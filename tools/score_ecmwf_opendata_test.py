#!/usr/bin/env python3
"""Score an ECMWF open-data GRIB run against local ERA5 with the validator metric.

Feasibility test: compares AIFS/IFS forecasts (hourly-interpolated onto the
Zeus grid) with ERA5 truth for one cycle, reporting latitude+Europe-weighted
RMSE/MAE per variable over the 0-48h and 0-360h challenge windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scoring import ValidatorFaithfulScorer
from evaluation.truth import Era5TruthLoader
from zeus.utils.coordinates import get_grid
from zeus.utils.region_mask import (
    OLD_REGION_CONFIGS,
    REGION_CONFIGS,
    build_geographic_weights,
)
from zeus.validator.constants import LATITUDE_WEIGHTS_PATH
from zeus_ml.features.era5_files import find_era5_files

SHORT_NAMES = {
    "2m_temperature": "2t",
    "100m_u_component_of_wind": "100u",
    "100m_v_component_of_wind": "100v",
    "surface_solar_radiation_downwards": "ssrd",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grib", required=True)
    parser.add_argument("--cycle", default="20260727T000000Z")
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--max-lead", type=int, default=360)
    parser.add_argument("--era5-root", default="data/evaluation/era5")
    parser.add_argument("--label", default="ecmwf")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(SHORT_NAMES),
        choices=list(SHORT_NAMES),
    )
    return parser


def load_forecast_steps(
    grib_path: str,
    short_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps_hours, fields) on the Zeus grid (lat ascending, lon -180..179.75)."""

    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"shortName": short_name},
            "indexpath": "",
        },
    )
    name = list(ds.data_vars)[0]
    da = ds[name]
    lats = np.asarray(da.latitude.values)
    lons = np.asarray(da.longitude.values)
    steps = (
        np.asarray(da.step.values).astype("timedelta64[h]").astype(int)
        if "step" in da.dims or "step" in da.coords
        else None
    )
    values = np.asarray(da.values, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
        steps = np.array([0])
    if lats[0] > lats[-1]:
        values = values[:, ::-1, :]
    if lons.max() > 180.0:
        # 0..359.75 -> -180..179.75
        values = np.roll(values, values.shape[-1] // 2, axis=-1)
    ds.close()
    order = np.argsort(steps)
    return steps[order], np.ascontiguousarray(values[order])


class WindowMetrics:
    def __init__(self, weights: torch.Tensor) -> None:
        self.weights = weights / weights.mean()
        self.sq = 0.0
        self.ab = 0.0
        self.count = 0

    def update(self, pred: torch.Tensor, truth: torch.Tensor) -> None:
        err = pred - truth
        self.sq += float((err.square() * self.weights).mean())
        self.ab += float((err.abs() * self.weights).mean())
        self.count += 1

    def result(self) -> dict[str, float]:
        rmse = float(np.sqrt(self.sq / self.count))
        mae = self.ab / self.count
        return {
            "iwRMSE": rmse,
            "iwMAE": mae,
            "combined": (rmse + mae) / 2.0,
        }


def main() -> int:
    args = build_parser().parse_args()
    cycle = datetime.strptime(args.cycle, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    latitude_weights = torch.from_numpy(np.load(LATITUDE_WEIGHTS_PATH)).to(
        torch.float32
    )
    grid = get_grid(-90.0, 90.0, -180.0, 179.75)
    regime = ValidatorFaithfulScorer.region_regime(cycle)
    configs = OLD_REGION_CONFIGS if regime == "europe_only" else REGION_CONFIGS
    geographic = build_geographic_weights(grid, configs)
    metric_weights = latitude_weights[:, None] * geographic

    truth_loader = Era5TruthLoader()
    results: dict[str, dict] = {}
    for variable in args.variables:
        short = SHORT_NAMES[variable]
        steps, fields = load_forecast_steps(args.grib, short)
        if variable == "surface_solar_radiation_downwards":
            # ECMWF ssrd is accumulated J/m2 since init; validator unit is
            # the previous-hour accumulation / 3600 (mean W/m2). Use the
            # containing inter-step interval's mean flux for each hour.
            interval_hours = np.diff(steps)
            flux = np.diff(fields, axis=0) / (
                interval_hours[:, None, None] * 3600.0
            )
            flux = np.clip(flux, 0.0, None)
        files = find_era5_files(args.era5_root, variable, cycle, args.max_lead)
        truth = truth_loader.load(
            files,
            variable=variable,
            cycle_time=cycle,
            horizon_hours=args.max_lead,
        ).tensor.to(torch.float32)

        short_metrics = WindowMetrics(metric_weights)
        long_metrics = WindowMetrics(metric_weights)
        max_step = int(steps.max())
        for lead in range(0, min(args.max_lead, max_step) + 1):
            if variable == "surface_solar_radiation_downwards":
                idx = min(
                    np.searchsorted(steps, lead, side="right") - 1,
                    flux.shape[0] - 1,
                )
                idx = max(idx, 0)
                pred = torch.from_numpy(flux[idx])
            else:
                idx = np.searchsorted(steps, lead, side="right") - 1
                idx = int(np.clip(idx, 0, len(steps) - 2))
                span = float(steps[idx + 1] - steps[idx])
                frac = (lead - float(steps[idx])) / span
                pred = torch.from_numpy(
                    (1.0 - frac) * fields[idx] + frac * fields[idx + 1]
                )
            if lead <= 48:
                short_metrics.update(pred, truth[lead])
            long_metrics.update(pred, truth[lead])
        results[variable] = {
            "window_0_48": short_metrics.result(),
            f"window_0_{min(args.max_lead, max_step)}": long_metrics.result(),
        }
        del fields, truth
        print(
            json.dumps({"variable": variable, **results[variable]}),
            flush=True,
        )

    print(
        json.dumps(
            {
                "label": args.label,
                "cycle": args.cycle,
                "regime": regime,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
