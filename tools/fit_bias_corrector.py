#!/usr/bin/env python3
"""Step 1 production: fit per-cell x hour-of-day bias maps from our own
published bundles (read-only) vs ERA5, validate on held-out cycles.

Usage (run when ERA5 covers the bundle dates, ~5 day lag):
  python tools/fit_bias_corrector.py                # auto split, fit + report
  python tools/fit_bias_corrector.py --holdout 4    # last 4 scoreable cycles held out

Writes bias maps to data/evaluation/bias_corrector/bias_v1.npz and a report
JSON next to it. Never writes to forecast_store_v2 or ecmwf_live.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path("/Zeus")
sys.path.insert(0, str(REPO))

from evaluation.scoring import ValidatorFaithfulScorer
from zeus.utils.compression import decompress_prediction
from zeus_ml.datasets.aifs_downscale_dataset import era5_to_zeus_grid

STORE = REPO / "data/forecast_store_v2/bundles"
ERA5 = REPO / "data/evaluation/era5"
OUTDIR = REPO / "data/evaluation/bias_corrector"
SHORT_HOURS = 49

VARS = [
    ("2m_temperature", "t2m"),
    ("100m_u_component_of_wind", "u100"),
    ("100m_v_component_of_wind", "v100"),
]

_day_cache: dict[tuple[str, str], np.ndarray | None] = {}


def era5_day(variable: str, code: str, day: str) -> np.ndarray | None:
    """(24,721,1440) float32 on the Zeus grid, or None if missing/NaN."""
    key = (variable, day)
    if key not in _day_cache:
        path = ERA5 / variable / f"era5_{day}.nc"
        if not path.is_file():
            _day_cache[key] = None
        else:
            with xr.open_dataset(path, engine="h5netcdf") as ds:
                da = ds[code]
                arr = np.asarray(da.values, np.float32)
            if not np.isfinite(arr).all():
                _day_cache[key] = None
            else:
                _day_cache[key] = np.stack(
                    [era5_to_zeus_grid(arr[h]) for h in range(arr.shape[0])]
                )
    return _day_cache[key]


def truth_for_cycle(variable: str, code: str, cycle: datetime,
                    hours: int) -> np.ndarray | None:
    fields = []
    for h in range(hours):
        valid = cycle + timedelta(hours=h)
        day = era5_day(variable, code, valid.strftime("%Y-%m-%d"))
        if day is None:
            return None
        fields.append(day[valid.hour])
    return np.stack(fields)


def load_short_artifact(cycle_key: str, variable: str) -> np.ndarray | None:
    path = STORE / cycle_key / f"{variable}@0_48.bin"
    if not path.is_file():
        return None
    tensor = decompress_prediction(path.read_bytes(), (SHORT_HOURS, 721, 1440))
    return tensor.numpy().astype(np.float32)


def cycle_dt(cycle_key: str) -> datetime:
    return datetime.strptime(cycle_key, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc)


def scoreable_cycles() -> list[str]:
    """Bundles whose full 49h window has finite ERA5 for all three variables."""
    good = []
    for key in sorted(p.name for p in STORE.iterdir() if p.is_dir()):
        cyc = cycle_dt(key)
        ok = True
        for variable, code in VARS:
            for h in (0, SHORT_HOURS - 1):  # cheap check: first and last day
                valid = cyc + timedelta(hours=h)
                if era5_day(variable, code, valid.strftime("%Y-%m-%d")) is None:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            good.append(key)
    return good


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=4)
    parser.add_argument("--shrink", type=float, default=0.75,
                        help="multiply hod bias by this factor at apply time")
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()

    cycles = scoreable_cycles()
    if len(cycles) < args.holdout + 4:
        print(f"Only {len(cycles)} scoreable cycles ({cycles}); need at least "
              f"{args.holdout + 4}. ERA5 has not caught up yet - rerun later.")
        return 1
    fit_keys, test_keys = cycles[: -args.holdout], cycles[-args.holdout:]
    print(f"fit on {len(fit_keys)} cycles {fit_keys[0]}..{fit_keys[-1]} | "
          f"holdout {test_keys}", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    scorer = ValidatorFaithfulScorer()
    bias_maps, report = {}, {"fit_cycles": fit_keys, "holdout": {},
                             "shrink": args.shrink}
    for variable, code in VARS:
        sums = np.zeros((24, 721, 1440), np.float64)
        counts = np.zeros(24, np.int64)
        for key in fit_keys:
            pred = load_short_artifact(key, variable)
            if pred is None:
                continue
            cyc = cycle_dt(key)
            truth = truth_for_cycle(variable, code, cyc, SHORT_HOURS)
            if truth is None:
                continue
            err = pred - truth
            for h in range(SHORT_HOURS):
                b = (cyc.hour + h) % 24
                sums[b] += err[h]
                counts[b] += 1
            print(f"  {variable}: accumulated {key}", flush=True)
        hod = (sums / np.maximum(counts, 1)[:, None, None]).astype(np.float32)
        bias_maps[variable] = hod

        rows = {}
        for key in test_keys:
            pred = load_short_artifact(key, variable)
            cyc = cycle_dt(key)
            truth = truth_for_cycle(variable, code, cyc, SHORT_HOURS)
            if pred is None or truth is None:
                continue
            idx = np.array([(cyc.hour + h) % 24 for h in range(SHORT_HOURS)])
            base = scorer.score(truth, pred, cycle_time=cyc, variable=variable)
            corr = scorer.score(truth, pred - args.shrink * hod[idx],
                                cycle_time=cyc, variable=variable)
            rows[key] = {
                "baseline": {"rmse": base.rmse, "mae": base.mae,
                             "combined": base.combined_error},
                "corrected": {"rmse": corr.rmse, "mae": corr.mae,
                              "combined": corr.combined_error},
            }
            print(f"{variable:36s} {key}  C {base.combined_error:.4f} -> "
                  f"{corr.combined_error:.4f}", flush=True)
        report["holdout"][variable] = rows

    np.savez_compressed(
        OUTDIR / f"bias_{args.tag}.npz",
        **{v: m for v, m in bias_maps.items()},
        fit_cycles=np.array(fit_keys),
        shrink=np.float32(args.shrink),
        created_utc=np.bytes_(datetime.now(timezone.utc).isoformat()),
    )
    (OUTDIR / f"bias_{args.tag}_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {OUTDIR / f'bias_{args.tag}.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
