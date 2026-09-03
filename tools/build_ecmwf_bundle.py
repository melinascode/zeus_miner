"""Build a serving-ready forecast bundle from a fetched ECMWF open-data run.

Reads the GRIBs written by tools/fetch_ecmwf_run.py and produces one .npz on
the Zeus grid (lat ascending -90..90, lon -180..179.75), float32:

  t2m   (61, 721, 1440)  K       ENS mean (em), step 0 from aifs-single
  u100  (61, 721, 1440)  m/s     ENS mean (cf + 50*pf)/51
  v100  (61, 721, 1440)  m/s     ENS mean
  ssrd  (60, 721, 1440)  W/m2    de-accumulated 6h interval mean; per interval
                                 the mean of available ensemble sources
                                 (AIFS-ENS 51, IFS-ENS 50), else aifs-single
  t2m_single/u100_single/v100_single (61, ...) deterministic fallback
  steps (61,) lead hours 0..360

If IFS-ENS wind GRIBs are present (ifs_ens_pf_winds_*.grib2), also:
  ifs_u100/ifs_v100 (n, 721, 1440) 50-member IFS-ENS wind means
  ifs_steps (n,) lead hours those means cover (06/18z runs stop at 144h)

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


def acc_to_intervals(
    acc_sum: dict[int, np.ndarray], counts: dict[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Member-mean accumulated ssrd -> (intervals (60,H,W) W/m2, available (60,))."""
    mean = {s: acc_sum[s] / counts[s] for s in acc_sum if counts.get(s)}
    mean.setdefault(0, np.zeros(SHAPE, np.float32))  # accumulation starts at 0
    intervals = np.zeros((len(STEPS) - 1, *SHAPE), np.float32)
    available = np.zeros(len(STEPS) - 1, bool)
    for i in range(len(STEPS) - 1):
        left, right = int(STEPS[i]), int(STEPS[i + 1])
        if left in mean and right in mean:
            intervals[i] = to_zeus_grid(
                (mean[right] - mean[left]) / (6 * 3600.0)
            )
            available[i] = True
    return np.clip(intervals, 0.0, None), available


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
    aifs_ssrd_sum: dict[int, np.ndarray] = {}
    aifs_ssrd_n: dict[int, int] = {}
    ifs_ssrd_sum: dict[int, np.ndarray] = {}
    ifs_ssrd_n: dict[int, int] = {}

    if not args.skip_ens:
        print("reading ens mean 2t (em) ...", flush=True)
        em = read_grib(os.path.join(run_dir, "aifs_ens_em_2t.grib2"), ["2t"])
        bundle["t2m"] = np.concatenate(
            [t2m_single[:1], stack(em["2t"], STEPS[1:])]
        )

        print("reading ens control winds + ssrd ...", flush=True)
        cf = read_grib(
            os.path.join(run_dir, "aifs_ens_cf_winds.grib2"),
            ["100u", "100v", "ssrd"],
        )
        pf_files = sorted(glob.glob(os.path.join(run_dir, "aifs_ens_pf_winds_*.grib2")))
        print(f"streaming perturbed members from {len(pf_files)} chunk files ...", flush=True)
        sums: dict[str, dict[int, np.ndarray]] = {"100u": {}, "100v": {}}
        for s, v in cf.get("ssrd", {}).items():
            aifs_ssrd_sum[s] = aifs_ssrd_sum.get(s, 0.0) + v
            aifs_ssrd_n[s] = aifs_ssrd_n.get(s, 0) + 1
        n_members = 0
        for pf_file in pf_files:
            chunk = read_grib(pf_file, ["100u", "100v", "ssrd"], average=False)
            lo, hi = map(int, os.path.basename(pf_file)[:-6].split("_")[-2:])
            n_members += hi - lo + 1
            for short in sums:
                for s, v in chunk[short].items():
                    sums[short][s] = sums[short].get(s, 0.0) + v
            for s, v in chunk.get("ssrd", {}).items():
                aifs_ssrd_sum[s] = aifs_ssrd_sum.get(s, 0.0) + v
                aifs_ssrd_n[s] = aifs_ssrd_n.get(s, 0) + (hi - lo + 1)
            print(f"  {os.path.basename(pf_file)} done", flush=True)
        for short, key in [("100u", "u100"), ("100v", "v100")]:
            mean = {
                s: (cf[short][s] + sums[short][s]) / (n_members + 1) for s in STEPS
            }
            bundle[key] = stack(mean, STEPS)
        print(f"ensemble mean over {n_members + 1} members (cf + pf)", flush=True)

    ifs_files = sorted(glob.glob(os.path.join(run_dir, "ifs_ens_pf_winds_*.grib2")))
    if ifs_files:
        print(f"streaming IFS-ENS winds + ssrd from {len(ifs_files)} chunk files ...", flush=True)
        ifs_sums: dict[str, dict[int, np.ndarray]] = {"100u": {}, "100v": {}}
        ifs_counts: dict[int, int] = {}
        for ifs_file in ifs_files:
            chunk = read_grib(ifs_file, ["100u", "100v", "ssrd"], average=False)
            lo, hi = map(int, os.path.basename(ifs_file)[:-6].split("_")[-2:])
            for short in ifs_sums:
                for s, v in chunk[short].items():
                    ifs_sums[short][s] = ifs_sums[short].get(s, 0.0) + v
            for s in chunk["100u"]:
                ifs_counts[s] = ifs_counts.get(s, 0) + (hi - lo + 1)
            for s, v in chunk.get("ssrd", {}).items():
                ifs_ssrd_sum[s] = ifs_ssrd_sum.get(s, 0.0) + v
                ifs_ssrd_n[s] = ifs_ssrd_n.get(s, 0) + (hi - lo + 1)
            print(f"  {os.path.basename(ifs_file)} done", flush=True)
        ifs_steps = np.array(sorted(ifs_counts), dtype=np.int64)
        n_ifs = ifs_counts[int(ifs_steps[0])]
        for short, key in [("100u", "ifs_u100"), ("100v", "ifs_v100")]:
            mean = {s: ifs_sums[short][s] / ifs_counts[s] for s in ifs_steps}
            bundle[key] = stack(mean, ifs_steps)
        bundle["ifs_steps"] = ifs_steps
        print(
            f"IFS-ENS wind mean over {n_ifs} members, "
            f"steps {int(ifs_steps[0])}..{int(ifs_steps[-1])}",
            flush=True,
        )

    ifs3h_files = sorted(glob.glob(os.path.join(run_dir, "ifs3h_ssrd_pf_*.grib2")))
    if ifs3h_files:
        print(f"streaming IFS-ENS 3h ssrd from {len(ifs3h_files)} chunk files ...",
              flush=True)
        acc_sum: dict[int, np.ndarray] = {}
        acc_n: dict[int, int] = {}
        for ifs3h_file in ifs3h_files:
            chunk = read_grib(ifs3h_file, ["ssrd"], average=False)
            lo, hi = map(int, os.path.basename(ifs3h_file)[:-6].split("_")[-2:])
            for s, v in chunk.get("ssrd", {}).items():
                acc_sum[s] = acc_sum.get(s, 0.0) + v
                acc_n[s] = acc_n.get(s, 0) + (hi - lo + 1)
            print(f"  {os.path.basename(ifs3h_file)} done", flush=True)
        steps_3h = np.arange(0, 61, 3)
        mean_3h = {s: acc_sum[s] / acc_n[s] for s in acc_sum}
        mean_3h.setdefault(0, np.zeros(SHAPE, np.float32))
        if all(int(s) in mean_3h for s in steps_3h):
            bundle["ifs_ssrd3h_acc"] = stack(mean_3h, steps_3h)
            bundle["ifs_ssrd3h_steps"] = steps_3h
            print(f"IFS 3h ssrd mean over {acc_n[3]} members, steps 0..60",
                  flush=True)
        else:
            missing = [int(s) for s in steps_3h if int(s) not in mean_3h]
            print(f"[warn] IFS 3h ssrd missing steps {missing}; skipping", flush=True)

    # SSRD: per 6h interval, mean of the available ensemble sources
    # (AIFS-ENS 51 members, IFS-ENS 50); intervals with no ensemble data
    # keep the aifs-single value. Wins -16..-18% 360h SSRD vs single.
    ssrd_sources = []
    if aifs_ssrd_n:
        ssrd_sources.append(("aifs_ens", *acc_to_intervals(aifs_ssrd_sum, aifs_ssrd_n)))
    if ifs_ssrd_n:
        ssrd_sources.append(("ifs_ens", *acc_to_intervals(ifs_ssrd_sum, ifs_ssrd_n)))
    if ssrd_sources:
        mixed = bundle["ssrd"].copy()
        ens_intervals = 0
        for i in range(mixed.shape[0]):
            fields = [ivals[i] for _, ivals, avail in ssrd_sources if avail[i]]
            if fields:
                mixed[i] = np.mean(fields, axis=0, dtype=np.float64).astype(np.float32)
                ens_intervals += 1
        bundle["ssrd"] = mixed
        ssrd_source = "+".join(name for name, _, _ in ssrd_sources)
        print(
            f"ssrd: ensemble mix ({ssrd_source}) on {ens_intervals}/"
            f"{mixed.shape[0]} intervals, single elsewhere",
            flush=True,
        )
    else:
        ssrd_source = "single"
        print("ssrd: no ensemble ssrd found, keeping aifs-single", flush=True)

    out_path = os.path.join(run_dir, "bundle.npz")
    np.savez(out_path, **bundle)
    meta = {
        "run": args.run,
        "primary": "aifs-ens-mean" if not args.skip_ens else "aifs-single",
        "downscaler_checkpoint": args.downscaler_checkpoint,
        "apply_downscaler": "hourly after linear interpolation of 6h steps",
        "ifs_wind_blend": bool(ifs_files),
        "ssrd_source": ssrd_source,
        "ifs_ssrd3h": "ifs_ssrd3h_acc" in bundle,
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
