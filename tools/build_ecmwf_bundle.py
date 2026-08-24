"""Build a serving-ready forecast bundle from a fetched ECMWF open-data run.

Reads the GRIBs written by tools/fetch_ecmwf_run.py and produces one .npz on
the Zeus grid (lat ascending -90..90, lon -180..179.75), float32:

  t2m   (61, 721, 1440)  K       ENS mean (em), step 0 from aifs-single
  u100  (61, 721, 1440)  m/s     ENS mean (cf + 50*pf)/51
  v100  (61, 721, 1440)  m/s     ENS mean
  ssrd  (60, 721, 1440)  W/m2    aifs-single, de-accumulated interval mean
  t2m_single/u100_single/v100_single (61, ...) deterministic fallback
  steps (61,) lead hours 0..360

Usage:
  python tools/build_ecmwf_bundle.py --run 20260818T120000Z
"""

import argparse
import glob
import json
import os

import eccodes
import numpy as np

RUN_ROOT = "/Zeus/data/evaluation/ecmwf_live"
STEPS = np.arange(0, 361, 6)
SHAPE = (721, 1440)


def to_zeus_grid(field: np.ndarray) -> np.ndarray:
    """ECMWF open data -> Zeus grid (lat -90..90, lon -180..180).

    These GRIBs already start at longitude 180 (i.e. -180), so only the
    latitude order differs from the Zeus convention.
    """
    return np.ascontiguousarray(field[::-1, :])


def read_grib(
    path: str, params: list[str], average: bool = True
) -> dict[str, dict[int, np.ndarray]]:
    """Stream a GRIB file, returning {param: {step: field}}.

    Multiple fields at the same (param, step) — e.g. ensemble members — are
    averaged when `average` is True, otherwise summed.
    """
    out: dict[str, dict[int, np.ndarray]] = {p: {} for p in params}
    counts: dict[str, dict[int, int]] = {p: {} for p in params}
    with open(path, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            short = eccodes.codes_get(gid, "shortName")
            if short in out:
                step = int(eccodes.codes_get(gid, "endStep"))
                vals = eccodes.codes_get_values(gid).reshape(SHAPE).astype(np.float32)
                if step in out[short]:
                    out[short][step] += vals
                    counts[short][step] += 1
                else:
                    out[short][step] = vals
                    counts[short][step] = 1
            eccodes.codes_release(gid)
    if average:
        for p in params:
            for step, n in counts[p].items():
                if n > 1:
                    out[p][step] /= n
    return out


def stack(fields: dict[int, np.ndarray], steps: np.ndarray) -> np.ndarray:
    missing = [s for s in steps if s not in fields]
    if missing:
        raise RuntimeError(f"missing steps {missing}")
    return np.stack([to_zeus_grid(fields[s]) for s in steps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. 20260818T120000Z")
    ap.add_argument("--skip-ens", action="store_true")
    ap.add_argument(
        "--downscaler-checkpoint",
        default="/Zeus/data/evaluation/training/aifs_downscaler_v2_ens.pt",
        help="Hourly residual CNN applied at serve time after interpolating 6h ENS.",
    )
    args = ap.parse_args()

    run_dir = os.path.join(RUN_ROOT, args.run)
    print("reading aifs-single ...", flush=True)
    single = read_grib(
        os.path.join(run_dir, "aifs_single_all.grib2"), ["2t", "100u", "100v", "ssrd"]
    )
    t2m_single = stack(single["2t"], STEPS)
    u100_single = stack(single["100u"], STEPS)
    v100_single = stack(single["100v"], STEPS)

    # ssrd: accumulated J/m2 since init -> mean W/m2 over each 6h interval
    ssrd_acc = stack(single["ssrd"], STEPS).astype(np.float64)
    ssrd = (np.diff(ssrd_acc, axis=0) / (6 * 3600.0)).astype(np.float32)
    ssrd = np.clip(ssrd, 0.0, None)

    bundle = {
        "steps": STEPS,
        "t2m_single": t2m_single,
        "u100_single": u100_single,
        "v100_single": v100_single,
        "ssrd": ssrd,
    }

    if not args.skip_ens:
        print("reading ens mean 2t (em) ...", flush=True)
        em = read_grib(os.path.join(run_dir, "aifs_ens_em_2t.grib2"), ["2t"])
        bundle["t2m"] = np.concatenate(
            [t2m_single[:1], stack(em["2t"], STEPS[1:])]
        )

        print("reading ens control winds ...", flush=True)
        cf = read_grib(os.path.join(run_dir, "aifs_ens_cf_winds.grib2"), ["100u", "100v"])
        pf_files = sorted(glob.glob(os.path.join(run_dir, "aifs_ens_pf_winds_*.grib2")))
        print(f"streaming perturbed members from {len(pf_files)} chunk files ...", flush=True)
        sums: dict[str, dict[int, np.ndarray]] = {"100u": {}, "100v": {}}
        n_members = 0
        for pf_file in pf_files:
            chunk = read_grib(pf_file, ["100u", "100v"], average=False)
            lo, hi = map(int, os.path.basename(pf_file)[:-6].split("_")[-2:])
            n_members += hi - lo + 1
            for short in sums:
                for s, v in chunk[short].items():
                    sums[short][s] = sums[short].get(s, 0.0) + v
            print(f"  {os.path.basename(pf_file)} done", flush=True)
        for short, key in [("100u", "u100"), ("100v", "v100")]:
            mean = {
                s: (cf[short][s] + sums[short][s]) / (n_members + 1) for s in STEPS
            }
            bundle[key] = stack(mean, STEPS)
        print(f"ensemble mean over {n_members + 1} members (cf + pf)", flush=True)

    out_path = os.path.join(run_dir, "bundle.npz")
    np.savez(out_path, **bundle)
    meta = {
        "run": args.run,
        "primary": "aifs-ens-mean" if not args.skip_ens else "aifs-single",
        "downscaler_checkpoint": args.downscaler_checkpoint,
        "apply_downscaler": "hourly after linear interpolation of 6h steps",
    }
    meta_path = os.path.join(run_dir, "bundle.meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"wrote {out_path} ({os.path.getsize(out_path)/1e9:.2f} GB)")
    print(f"wrote {meta_path}")
    for k, v in bundle.items():
        if hasattr(v, "shape"):
            print(f"  {k:12s} {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
