#!/usr/bin/env python3
"""Fetch ERA5 NetCDF files into data/evaluation/era5 only.

Never writes into the validator ERA5 cache.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = Path("data/evaluation/era5")
VARIABLES = (
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download CDS ERA5 single-level daily NetCDF files into "
            "data/evaluation/era5/{variable}/."
        )
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--variable",
        action="append",
        choices=VARIABLES,
        help="Repeatable. Defaults to all four Zeus variables.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without contacting CDS.",
    )
    parser.add_argument(
        "--env-file",
        default="validator.env",
        help="Optional env file containing CDS_API_KEY (never printed).",
    )
    return parser


def _load_cds_api_key(env_file: str | Path) -> str:
    key = os.environ.get("CDS_API_KEY", "").strip()
    if key:
        return key
    path = Path(env_file)
    if not path.is_file():
        path = PROJECT_ROOT / env_file
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "CDS_API_KEY":
                key = value.strip().strip("'").strip('"')
                if key:
                    return key
    raise SystemExit(
        "CDS_API_KEY not found in environment or env file. "
        "Set CDS_API_KEY or pass --env-file pointing at validator.env."
    )


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if "data/evaluation" not in output_dir.as_posix():
        raise SystemExit(
            "ERA5 evaluation downloads must live under data/evaluation/."
        )

    start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    end = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    if end < start:
        raise SystemExit("--end-date must be on or after --start-date.")

    variables = tuple(args.variable) if args.variable else VARIABLES
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)

    planned = [
        output_dir / variable / f"era5_{day.strftime('%Y-%m-%d')}.nc"
        for variable in variables
        for day in days
    ]
    missing = [path for path in planned if not path.is_file()]
    print(
        {
            "output_dir": str(output_dir),
            "variables": list(variables),
            "days": len(days),
            "planned_files": len(planned),
            "missing_files": len(missing),
            "dry_run": args.dry_run,
        },
        flush=True,
    )
    if args.dry_run:
        for path in missing[:20]:
            print(f"MISSING {path}", flush=True)
        if len(missing) > 20:
            print(f"... and {len(missing) - 20} more", flush=True)
        return 0

    cds_api_key = _load_cds_api_key(args.env_file)
    import cdsapi
    from zeus.validator.constants import COPERNICUS_ERA5_URL

    client = cdsapi.Client(
        url=COPERNICUS_ERA5_URL,
        key=cds_api_key,
        quiet=True,
        progress=False,
        warning_callback=lambda _: None,
        sleep_max=10,
    )
    client.warning_callback = None

    failures = 0
    for variable in variables:
        for day in days:
            destination = (
                output_dir / variable / f"era5_{day.strftime('%Y-%m-%d')}.nc"
            )
            if destination.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            request = {
                "product_type": ["reanalysis"],
                "variable": [variable],
                "year": [str(day.year)],
                "month": [f"{day.month:02d}"],
                "day": [f"{day.day:02d}"],
                "time": [f"{hour:02d}:00" for hour in range(24)],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            temporary = destination.with_suffix(".nc.tmp")
            try:
                client.retrieve(
                    "reanalysis-era5-single-levels",
                    request,
                    target=str(temporary),
                )
                temporary.replace(destination)
                print(f"DOWNLOADED {destination}", flush=True)
            except Exception as exc:  # noqa: BLE001 - continue batch
                failures += 1
                temporary.unlink(missing_ok=True)
                print(
                    f"FAILED {destination}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
    print({"failures": failures}, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
