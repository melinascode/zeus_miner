#!/usr/bin/env python3
"""Materialize evaluation ERA5 NetCDF files from Google ARCO ERA5.

Used when CDS credentials are unavailable. Writes only under
data/evaluation/era5/. Produces unit-annotated files compatible with
evaluation.truth.Era5TruthLoader (same product family as CDS ERA5).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = Path("data/evaluation/era5")
DEFAULT_ZARR = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
VARIABLES = (
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
)
UNITS = {
    "2m_temperature": "K",
    "100m_u_component_of_wind": "m s**-1",
    "100m_v_component_of_wind": "m s**-1",
    "surface_solar_radiation_downwards": "J m**-2",
}
SHORT_CODES = {
    "2m_temperature": "t2m",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "surface_solar_radiation_downwards": "ssrd",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--variable", action="append", choices=VARIABLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zarr-url", default=DEFAULT_ZARR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if "data/evaluation" not in output_dir.as_posix():
        raise SystemExit("ERA5 evaluation files must live under data/evaluation/.")

    import numpy as np
    import xarray as xr

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    variables = tuple(args.variable) if args.variable else VARIABLES

    print(f"Opening {args.zarr_url}", flush=True)
    dataset = xr.open_zarr(
        args.zarr_url,
        chunks=None,
        storage_options={"token": "anon"},
    )
    missing = [name for name in variables if name not in dataset]
    if missing:
        raise SystemExit(f"Zarr missing variables: {missing}")

    day = start
    failures = 0
    while day <= end:
        for variable in variables:
            destination = (
                output_dir / variable / f"era5_{day.isoformat()}.nc"
            )
            if destination.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                day_slice = dataset[variable].sel(
                    time=slice(
                        f"{day.isoformat()}T00:00",
                        f"{day.isoformat()}T23:00",
                    )
                )
                # Load into memory for this day only.
                values = day_slice.load()
                if values.sizes.get("time", 0) != 24:
                    raise ValueError(
                        f"Expected 24 hours for {day}, got "
                        f"{values.sizes.get('time')}"
                    )
                short = SHORT_CODES[variable]
                # Match CDS-like coordinate orientation used by truth loader:
                # latitude descending in file is OK (loader sorts), longitude
                # can be 0..359.75 (loader normalizes).
                out = values.to_dataset(name=short)
                if "time" in out.dims:
                    out = out.rename({"time": "valid_time"})
                out[short].attrs["units"] = UNITS[variable]
                out.attrs["source"] = "google-arco-era5"
                out.attrs["zarr_url"] = args.zarr_url
                temporary = destination.with_suffix(".nc.tmp")
                out.to_netcdf(temporary, engine="h5netcdf")
                temporary.replace(destination)
                print(f"DOWNLOADED {destination}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(
                    f"FAILED {destination}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        day = day + timedelta(days=1)

    print({"failures": failures}, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
