"""Score IFS-ENS blends against the current serving cube on one cycle.

Candidates (all on official validator weights, iwRMSE/iwMAE per variable):
  serving        hourly v2 cube (AIFS-ENS mean + downscaler v2)  <- baseline
  aifs_ens       raw AIFS-ENS mean, linear hourly interpolation
  ifs_ens        raw IFS-ENS mean (50 pf members), linear interpolation
  ifs_det        deterministic IFS oper run
  ens_blend50    0.5*aifs_ens + 0.5*ifs_ens (all three variables)
  serve_wb35     cube, winds replaced by 0.65*cube + 0.35*ifs_ens
  serve_wb50     cube, winds replaced by 0.50*cube + 0.50*ifs_ens
  serve_wb65     cube, winds replaced by 0.35*cube + 0.65*ifs_ens

Inputs:
  --cube-dir   directory with hourly_tuv_f16.npy for the cycle
  --ifs-mean   /Zeus/data/evaluation/ifs_ens_mean/<cycle>.npy  (fetch_ifs_ens_means.py)
  --ifs-oper   GRIB of the deterministic run (kept by the same fetcher)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from zeus_ml.datasets.aifs_downscale_dataset import EnsMeanCycleReader, Era5HourlyReader
from zeus_ml.evaluate.evaluate_aifs_downscaler import SHORT_NAMES, Stream, cycle_maps
from zeus_ml.models.aifs_downscaler_cnn import MAX_LEAD_HOURS, STEP_HOURS

WINDOWS = (12, 24, 48, 72, 120, 360)


def interpolate_regular(fields: np.ndarray, lead: int, step: int = STEP_HOURS) -> np.ndarray:
    last = (fields.shape[0] - 1) * step
    if lead >= last:
        return fields[-1].astype(np.float32)
    left = lead // step
    fraction = (lead - left * step) / float(step)
    return (1.0 - fraction) * fields[left].astype(np.float32) + fraction * fields[
        left + 1
    ].astype(np.float32)


def decode_oper(path: Path) -> np.ndarray:
    """Deterministic run -> (61, 3, 721, 1440) float32, steps 0..360 by 6."""
    import eccodes

    index = {"2t": 0, "100u": 1, "100v": 2}
    out = np.full((MAX_LEAD_HOURS // STEP_HOURS + 1, 3, 721, 1440), np.nan, np.float32)
    with open(path, "rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                break
            short = eccodes.codes_get(gid, "shortName")
            step = int(eccodes.codes_get(gid, "endStep"))
            if short in index and step % STEP_HOURS == 0 and step <= MAX_LEAD_HOURS:
                values = eccodes.codes_get_values(gid).reshape(721, 1440)
                out[step // STEP_HOURS, index[short]] = values[::-1, :]
            eccodes.codes_release(gid)
    if not np.isfinite(out).all():
        raise ValueError(f"{path.name}: missing fields")
    return out


def fmt(final: list[dict]) -> str:
    return "  ".join(
        f"{name}={final[i]['rmse']:.3f}/{final[i]['mae']:.3f}"
        for i, name in enumerate(SHORT_NAMES)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--cube-dir", required=True)
    parser.add_argument("--ifs-mean", default=None)
    parser.add_argument("--ifs-oper", default=None)
    parser.add_argument("--ens-root", default="/Zeus/data/evaluation/aifs_ens_mean")
    parser.add_argument("--era5-root", default="/Zeus/data/evaluation/era5")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    cycle = args.cycle
    ifs_mean_path = args.ifs_mean or f"/Zeus/data/evaluation/ifs_ens_mean/{cycle}.npy"
    ifs_oper_path = args.ifs_oper or (
        f"/Zeus/data/evaluation/ifs_ens/{cycle}/ifs_oper_all.grib2"
    )
    cycle_time = datetime.strptime(cycle, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    latitudes = torch.linspace(-90.0, 90.0, 721)
    _, weights = cycle_maps(cycle_time, latitudes)
    truth = Era5HourlyReader(args.era5_root)

    cube = np.load(Path(args.cube_dir) / "hourly_tuv_f16.npy", mmap_mode="r")
    aifs_fields = EnsMeanCycleReader(args.ens_root, cache_size=1).get(cycle)
    ifs_fields = np.load(ifs_mean_path)
    print(f"decoding deterministic IFS {ifs_oper_path}", flush=True)
    det_fields = decode_oper(Path(ifs_oper_path))

    methods = (
        "serving",
        "aifs_ens",
        "ifs_ens",
        "ifs_det",
        "ens_blend50",
        "serve_wb35",
        "serve_wb50",
        "serve_wb65",
        "serve_ramp35",
        "serve_ramp50",
        "serve_ramp65",
    )
    streams = {name: {h: Stream() for h in WINDOWS} for name in methods}

    for lead in range(MAX_LEAD_HOURS + 1):
        serving = cube[lead].astype(np.float32)
        aifs = interpolate_regular(aifs_fields, lead)
        ifs = interpolate_regular(ifs_fields, lead)
        det = interpolate_regular(det_fields, lead)
        truth_hour = torch.from_numpy(truth.read(cycle_time + timedelta(hours=lead)))

        def wind_blend(alpha: float) -> np.ndarray:
            out = serving.copy()
            out[1:] = (1.0 - alpha) * serving[1:] + alpha * ifs[1:]
            return out

        # Ramp: no IFS below 72h (protects the 49h challenge), linear to
        # alpha_max at 360h where IFS-ENS pulls ahead.
        ramp = max(0.0, min(1.0, (lead - 72) / (360.0 - 72.0)))

        preds = {
            "serving": serving,
            "aifs_ens": aifs,
            "ifs_ens": ifs,
            "ifs_det": det,
            "ens_blend50": 0.5 * aifs + 0.5 * ifs,
            "serve_wb35": wind_blend(0.35),
            "serve_wb50": wind_blend(0.50),
            "serve_wb65": wind_blend(0.65),
            "serve_ramp35": wind_blend(0.35 * ramp),
            "serve_ramp50": wind_blend(0.50 * ramp),
            "serve_ramp65": wind_blend(0.65 * ramp),
        }
        for name, pred in preds.items():
            tensor = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32))
            for horizon, stream in streams[name].items():
                if lead <= horizon:
                    stream.update(tensor, truth_hour, weights)
        if lead % 60 == 0:
            print(f"  lead {lead}", flush=True)

    print(f"\n{cycle}  iwRMSE/iwMAE per variable")
    for horizon in WINDOWS:
        print(f"  {horizon}h")
        for name in methods:
            print(f"    {name:12s} {fmt(streams[name][horizon].finalize())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
