#!/usr/bin/env python3
"""Official-scalar iwRMSE/iwMAE split: total / Europe / Germany.

Scores a serving cube vs ERA5. Weights are cosine-lat × official capacity
scalar, then optionally masked to the Europe or Germany validator box.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from zeus.utils.coordinates import get_grid
from zeus.utils.region_mask import geographic_scalar_for_variable, region_masks_for_grid
from zeus.validator.constants import LATITUDE_WEIGHTS_PATH
from zeus_ml.datasets.aifs_downscale_dataset import era5_to_zeus_grid

VARS = (
    ("2m_temperature", "t2m", "tuv", 0, False),
    ("100m_u_component_of_wind", "u100", "tuv", 1, False),
    ("100m_v_component_of_wind", "v100", "tuv", 2, False),
    ("surface_solar_radiation_downwards", "ssrd", "ssrd", None, True),
)
REGIONS = ("total", "europe", "germany")
HORIZONS = (48, 360)


def latitude_weights() -> np.ndarray:
    return np.load(LATITUDE_WEIGHTS_PATH).astype(np.float32)


def weight_maps() -> dict[str, dict[str, np.ndarray]]:
    lat = latitude_weights()[:, None]
    masks = region_masks_for_grid(get_grid(-90.0, 90.0, -180.0, 179.75))
    out: dict[str, dict[str, np.ndarray]] = {}
    for variable, *_ in VARS:
        base = (lat * geographic_scalar_for_variable(variable).numpy()).astype(
            np.float32
        )
        out[variable] = {
            "total": base,
            "europe": base * masks["europe"].numpy(),
            "germany": base * masks["germany"].numpy(),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", default="20260801T000000Z")
    parser.add_argument(
        "--cube",
        default="/Zeus/data/evaluation/scoring_cubes/20260801T000000Z_v3",
    )
    parser.add_argument("--era5-root", default="/Zeus/data/evaluation/era5")
    parser.add_argument(
        "--output",
        default="/Zeus/data/evaluation/training/official_region_split_20260801.json",
    )
    args = parser.parse_args()

    cycle_time = datetime.strptime(args.cycle, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    cube = Path(args.cube)
    tuv = np.load(cube / "hourly_tuv_f16.npy", mmap_mode="r")
    ssrd = np.load(cube / "hourly_ssrd_f16.npy", mmap_mode="r")
    weights = weight_maps()
    mass = {
        var: {r: float(w.sum()) for r, w in maps.items()}
        for var, maps in weights.items()
    }
    print("official scalar mass shares (of total):", flush=True)
    for var, maps in mass.items():
        tot = maps["total"]
        print(
            f"  {var:40s} DE={maps['germany']/tot:.3f}  "
            f"EU={maps['europe']/tot:.3f}  rest={1-maps['europe']/tot:.3f}",
            flush=True,
        )

    acc = {
        var: {
            r: {h: [0.0, 0.0, 0] for h in HORIZONS} for r in REGIONS
        }
        for var, *_ in VARS
    }
    era5 = Path(args.era5_root)
    open_ds: dict = {}

    def truth(variable: str, code: str, valid: datetime, solar: bool) -> np.ndarray:
        day = valid.strftime("%Y-%m-%d")
        key = (variable, day)
        if key not in open_ds:
            ds = xr.open_dataset(era5 / variable / f"era5_{day}.nc", engine="h5netcdf")
            if len(open_ds) > 6:
                open_ds.pop(next(iter(open_ds))).close()
            open_ds[key] = ds
        values = open_ds[key][code].isel({open_ds[key][code].dims[0]: valid.hour}).values
        field = era5_to_zeus_grid(np.asarray(values, dtype=np.float32))
        if solar:
            field = np.clip(field / 3600.0, 0.0, None)
        return field

    started = datetime.now(timezone.utc)
    for lead in range(361):
        valid = cycle_time + timedelta(hours=lead)
        for variable, code, kind, channel, solar in VARS:
            pred = (
                np.asarray(ssrd[lead], dtype=np.float32)
                if kind == "ssrd"
                else np.asarray(tuv[lead, channel], dtype=np.float32)
            )
            err = pred - truth(variable, code, valid, solar)
            sq = err * err
            ab = np.abs(err)
            for region, w in weights[variable].items():
                sse = float((sq * w).sum())
                sae = float((ab * w).sum())
                m = mass[variable][region]
                for h in HORIZONS:
                    if lead <= h:
                        acc[variable][region][h][0] += sse / m
                        acc[variable][region][h][1] += sae / m
                        acc[variable][region][h][2] += 1
        if lead % 48 == 0:
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            print(f"lead {lead}/360  {elapsed:.0f}s", flush=True)

    packed = {}
    print(
        f"\n{args.cycle}  cube={cube.name}",
        flush=True,
    )
    for h in HORIZONS:
        print(f"\n-- {h}h  official-scalar iwRMSE/iwMAE --", flush=True)
        packed[str(h)] = {}
        for variable, *_ in VARS:
            packed[str(h)][variable] = {}
            row = []
            for region in REGIONS:
                n = acc[variable][region][h][2]
                rmse = (acc[variable][region][h][0] / n) ** 0.5
                mae = acc[variable][region][h][1] / n
                packed[str(h)][variable][region] = {
                    "rmse": rmse,
                    "mae": mae,
                    "combined": (rmse + mae) / 2.0,
                }
                row.append(f"{region}={rmse:.3f}/{mae:.3f}")
            print(f"  {variable:40s}  " + "  ".join(row), flush=True)

    payload = {
        "cycle": args.cycle,
        "cube": str(cube),
        "weights": "cosine_lat * official_capacity_scalar, masked to region",
        "mass_share_of_total": {
            var: {r: maps[r] / maps["total"] for r in REGIONS}
            for var, maps in mass.items()
        },
        "horizons": packed,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
