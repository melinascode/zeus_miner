"""Score ENS linear / global CNN / composed specialists on official maps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from zeus.utils.region_mask import geographic_scalar_for_variable, region_masks_for_grid
from zeus.utils.coordinates import get_grid
from zeus_ml.datasets.aifs_downscale_dataset import Era5HourlyReader
from zeus_ml.models.aifs_downscaler_cnn import MAX_LEAD_HOURS
from zeus_ml.serve.compose_forecast import ComposerConfig, ForecastComposer


class MassStream:
    def __init__(self, n_vars: int = 3) -> None:
        self.squared = torch.zeros(n_vars, dtype=torch.float64)
        self.absolute = torch.zeros(n_vars, dtype=torch.float64)
        self.mass = 0.0

    def update(self, prediction, truth, weights) -> None:
        error = prediction - truth
        w = weights.unsqueeze(0)
        self.squared += (error.square() * w).sum(dim=(-2, -1)).double()
        self.absolute += (error.abs() * w).sum(dim=(-2, -1)).double()
        self.mass += float(weights.sum())

    def finalize(self) -> list[dict[str, float]]:
        denom = max(self.mass, 1e-18)
        mse = self.squared / denom
        mae = self.absolute / denom
        rmse = torch.sqrt(mse)
        return [
            {
                "rmse": float(rmse[i]),
                "mae": float(mae[i]),
                "combined": float((rmse[i] + mae[i]) / 2.0),
            }
            for i in range(3)
        ]

SHORT = ("2t", "100u", "100v")
METHODS = ("ens_linear", "global_cnn", "composed_add", "composed_replace")
HORIZONS = (48, 360)


def official_maps(latitudes: torch.Tensor) -> dict[str, torch.Tensor]:
    cosine = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)[:, None]
    grid = get_grid(-90.0, 90.0, -180.0, 179.75)
    masks = region_masks_for_grid(grid)
    temp = cosine * geographic_scalar_for_variable("2m_temperature")
    wind = cosine * geographic_scalar_for_variable("100m_u_component_of_wind")
    return {
        "official_temp": temp,
        "official_wind": wind,
        "germany": cosine * masks["germany"],
        "europe": cosine * masks["europe"],
    }


def fmt(rows: list[dict]) -> str:
    return "  ".join(
        f"{n}={rows[i]['rmse']:.3f}/{rows[i]['mae']:.3f}" for i, n in enumerate(SHORT)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", action="append", default=[])
    parser.add_argument("--era5-root", default="/Zeus/data/evaluation/era5")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        default="/Zeus/data/evaluation/training/composed_forecast_scores.json",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cycles = args.cycle or ["20260727T000000Z", "20260801T000000Z"]
    composer = ForecastComposer(ComposerConfig())
    latitudes = composer.latitudes
    weights = official_maps(latitudes)
    truth = Era5HourlyReader(args.era5_root, cache_size=4)

    report: dict = {"cycles": []}
    for cycle in cycles:
        cycle_time = datetime.strptime(cycle, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
        streams = {
            method: {
                name: {h: MassStream() for h in HORIZONS} for name in weights
            }
            for method in METHODS
        }
        print(f"{cycle}", flush=True)
        for lead in range(MAX_LEAD_HOURS + 1):
            fields = composer.predict_hour(cycle, lead)
            truth_hour = torch.from_numpy(
                truth.read(cycle_time + timedelta(hours=lead))
            )
            for method in METHODS:
                pred = fields[method]
                for wname, w in weights.items():
                    for horizon in HORIZONS:
                        if lead <= horizon:
                            streams[method][wname][horizon].update(pred, truth_hour, w)
            if lead in (0, 48, 120, 240, 360) or lead % 60 == 0:
                print(f"  lead {lead}", flush=True)
        cycle_row = {"cycle": cycle, "horizons": {}}
        for horizon in HORIZONS:
            print(f"  -- {horizon}h --")
            cycle_row["horizons"][str(horizon)] = {}
            for method in METHODS:
                cycle_row["horizons"][str(horizon)][method] = {}
                for wname in weights:
                    rows = streams[method][wname][horizon].finalize()
                    cycle_row["horizons"][str(horizon)][method][wname] = rows
                    print(f"    {method:12s} {wname:14s} {fmt(rows)}")
        report["cycles"].append(cycle_row)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
