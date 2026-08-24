#!/usr/bin/env python3
"""Write 721x1440 land-sea and orography maps for v4 static channels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ZARR = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
DEFAULT_OUTPUT = Path("data/evaluation/training/v4_static")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zarr-url", default=DEFAULT_ZARR)
    return parser


def _align_lat_lon(values, latitudes, longitudes):
    import numpy as np

    latitude = np.asarray(latitudes, dtype=np.float64)
    longitude = np.asarray(longitudes, dtype=np.float64)
    if latitude[0] > latitude[-1]:
        values = np.flip(values, axis=0)
        latitude = latitude[::-1]
    if longitude.min() >= 0:
        shift = int(np.argmin(np.abs(longitude - 180.0)))
        values = np.roll(values, -shift, axis=1)
        longitude = (longitude + 180.0) % 360.0 - 180.0
        order = np.argsort(longitude)
        values = values[:, order]
        longitude = longitude[order]
    target_lat = np.linspace(-90.0, 90.0, 721)
    target_lon = np.arange(-180.0, 180.0, 0.25)
    if not np.allclose(latitude, target_lat, atol=1e-4):
        raise ValueError("Latitude grid does not match Zeus 721-point mesh.")
    if not np.allclose(longitude, target_lon, atol=1e-4):
        raise ValueError("Longitude grid does not match Zeus 1440-point mesh.")
    return np.ascontiguousarray(values, dtype=np.float32)


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if "data/evaluation" not in output_dir.as_posix():
        raise SystemExit("v4 static maps must live under data/evaluation/.")
    output_dir.mkdir(parents=True, exist_ok=True)

    import xarray as xr

    print(f"Opening {args.zarr_url}", flush=True)
    dataset = xr.open_zarr(
        args.zarr_url,
        chunks=None,
        storage_options={"token": "anon"},
    )
    land_name = "land_sea_mask" if "land_sea_mask" in dataset else None
    oro_name = (
        "geopotential_at_surface"
        if "geopotential_at_surface" in dataset
        else None
    )
    if land_name is None or oro_name is None:
        raise SystemExit(
            f"Zarr missing static fields. Have: {list(dataset.data_vars)[:20]}"
        )
    land = dataset[land_name]
    orography = dataset[oro_name]
    if "time" in land.dims:
        land = land.sel(time="2025-07-01T00:00")
    if "time" in orography.dims:
        orography = orography.sel(time="2025-07-01T00:00")
    land_values = _align_lat_lon(
        land.load().values,
        land.latitude.values,
        land.longitude.values,
    )
    oro_values = _align_lat_lon(
        orography.load().values,
        orography.latitude.values,
        orography.longitude.values,
    )
    # Convert surface geopotential to kilometers, then a bounded scale.
    oro_km = oro_values / np.float32(9.80665 * 1000.0)
    land_values = np.clip(land_values, 0.0, 1.0)
    np.save(output_dir / "land_sea.npy", land_values)
    np.save(output_dir / "orography.npy", oro_km.astype(np.float32, copy=False))
    print(
        {
            "land_sea": str(output_dir / "land_sea.npy"),
            "orography": str(output_dir / "orography.npy"),
            "land_mean": float(land_values.mean()),
            "oro_min": float(oro_km.min()),
            "oro_max": float(oro_km.max()),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
