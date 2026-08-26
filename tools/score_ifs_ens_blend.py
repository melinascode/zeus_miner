"""Score IFS, AIFS-ENS mean, and lead-ramped blends on one cycle.

Reports the same 48h / 360h iwRMSE / iwMAE as tools/score_aifs_cycle.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import eccodes
import numpy as np
import torch

from zeus_ml.datasets.aifs_downscale_dataset import (
    GRIB_SHORT_NAMES,
    EnsMeanCycleReader,
    Era5HourlyReader,
    to_zeus_grid,
)
from zeus_ml.evaluate.evaluate_aifs_downscaler import SHORT_NAMES, Stream, cycle_maps
from zeus_ml.models.aifs_downscaler_cnn import (
    FULL_HEIGHT,
    FULL_WIDTH,
    MAX_LEAD_HOURS,
    STEP_HOURS,
)


def fmt(final: list[dict]) -> str:
    return "  ".join(
        f"{name}={final[i]['rmse']:.3f}/{final[i]['mae']:.3f}"
        for i, name in enumerate(SHORT_NAMES)
    )


def ifs_weight(lead: int, hold: int, fade: int) -> float:
    if lead <= hold:
        return 1.0
    if lead >= fade:
        return 0.0
    return 1.0 - (lead - hold) / (fade - hold)


def decode_ifs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, fields) with fields shape (nstep, 3, 721, 1440) float32."""

    index = {name: i for i, name in enumerate(GRIB_SHORT_NAMES)}
    buckets: dict[int, np.ndarray] = {}
    seen: dict[int, np.ndarray] = {}
    with open(path, "rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                break
            short = eccodes.codes_get(gid, "shortName")
            step = int(eccodes.codes_get(gid, "endStep"))
            if short in index:
                if step not in buckets:
                    buckets[step] = np.zeros(
                        (len(GRIB_SHORT_NAMES), FULL_HEIGHT, FULL_WIDTH), np.float32
                    )
                    seen[step] = np.zeros(len(GRIB_SHORT_NAMES), bool)
                values = eccodes.codes_get_values(gid).reshape(FULL_HEIGHT, FULL_WIDTH)
                buckets[step][index[short]] = to_zeus_grid(values)
                seen[step][index[short]] = True
            eccodes.codes_release(gid)
    incomplete = [step for step, mask in seen.items() if not mask.all()]
    if incomplete:
        raise ValueError(f"{path.name} incomplete at steps {incomplete[:8]}")
    steps = np.array(sorted(buckets), dtype=np.int32)
    fields = np.stack([buckets[int(step)] for step in steps], axis=0)
    return steps, fields


def interpolate_steps(
    steps: np.ndarray, fields: np.ndarray, lead: int
) -> np.ndarray:
    if lead <= int(steps[0]):
        return fields[0]
    if lead >= int(steps[-1]):
        return fields[-1]
    right = int(np.searchsorted(steps, lead, side="left"))
    if int(steps[right]) == lead:
        return fields[right]
    left = right - 1
    span = float(steps[right] - steps[left])
    fraction = (lead - float(steps[left])) / span
    return (1.0 - fraction) * fields[left] + fraction * fields[right]


def interpolate_regular(fields: np.ndarray, lead: int, step: int = STEP_HOURS) -> np.ndarray:
    last = (fields.shape[0] - 1) * step
    if lead >= last:
        return fields[-1].astype(np.float32)
    left = lead // step
    fraction = (lead - left * step) / float(step)
    return (1.0 - fraction) * fields[left].astype(np.float32) + fraction * fields[
        left + 1
    ].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", default="20260801T000000Z")
    parser.add_argument(
        "--ifs-grib",
        default="/Zeus/data/evaluation/ecmwf_test/ifs_20260801_00z.grib2",
    )
    parser.add_argument("--ens-root", default="/Zeus/data/evaluation/aifs_ens_mean")
    parser.add_argument("--era5-root", default="/Zeus/data/evaluation/era5")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    cycle = args.cycle
    cycle_time = datetime.strptime(cycle, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    latitudes = torch.linspace(-90.0, 90.0, 721)
    _, weights = cycle_maps(cycle_time, latitudes)
    truth = Era5HourlyReader(args.era5_root)

    print(f"decoding IFS {args.ifs_grib}", flush=True)
    ifs_steps, ifs_fields = decode_ifs(Path(args.ifs_grib))
    print(f"  IFS steps {len(ifs_steps)}  {int(ifs_steps[0])}..{int(ifs_steps[-1])}", flush=True)
    ens_fields = EnsMeanCycleReader(args.ens_root, cache_size=1).get(cycle)
    print(f"  ENS {ens_fields.shape}", flush=True)

    schedules = {
        "IFS": None,
        "ENS-mean": None,
        "blend 48->120": (48, 120),
        "blend 72->120": (72, 120),
        "blend 48->72": (48, 72),
    }
    windows = (12, 24, 48, 72, 120, 360)
    streams = {name: {h: Stream() for h in windows} for name in schedules}

    for lead in range(MAX_LEAD_HOURS + 1):
        ifs = interpolate_steps(ifs_steps, ifs_fields, lead)
        ens = interpolate_regular(ens_fields, lead)
        truth_hour = torch.from_numpy(truth.read(cycle_time + timedelta(hours=lead)))
        preds = {
            "IFS": ifs,
            "ENS-mean": ens,
            "blend 48->120": ifs_weight(lead, 48, 120) * ifs
            + (1.0 - ifs_weight(lead, 48, 120)) * ens,
            "blend 72->120": ifs_weight(lead, 72, 120) * ifs
            + (1.0 - ifs_weight(lead, 72, 120)) * ens,
            "blend 48->72": ifs_weight(lead, 48, 72) * ifs
            + (1.0 - ifs_weight(lead, 48, 72)) * ens,
        }
        for name, pred in preds.items():
            tensor = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32))
            for horizon, stream in streams[name].items():
                if lead <= horizon:
                    stream.update(tensor, truth_hour, weights)
        if lead in (48, 360) or lead % 120 == 0:
            print(f"  lead {lead}", flush=True)

    print(f"{cycle} IFS vs ENS-mean vs blends")
    for horizon in windows:
        print(f"  {horizon}h")
        for name in schedules:
            print(f"    {name:16s} {fmt(streams[name][horizon].finalize())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
