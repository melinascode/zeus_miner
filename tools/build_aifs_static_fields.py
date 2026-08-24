"""Write 721x1440 land-sea and orography maps from AIFS open data.

AIFS publishes surface geopotential (z) and the land-sea mask (lsm) as
time-invariant fields, so taking them from the same model that produces the
forecasts keeps the static channels on exactly the input grid.

Usage:
  python tools/build_aifs_static_fields.py --date 2026-04-30
"""

from __future__ import annotations

import argparse
import os
import tempfile

import eccodes
import numpy as np
from ecmwf.opendata import Client

GRAVITY = 9.80665
SHAPE = (721, 1440)
DEFAULT_OUTPUT = "/Zeus/data/evaluation/training/aifs_static"


def to_zeus_grid(field: np.ndarray) -> np.ndarray:
    """ECMWF open data -> Zeus grid (lat -90..90, lon -180..180).

    These GRIBs already start at longitude 180 (i.e. -180), so only the
    latitude order differs from the Zeus convention.
    """
    return np.ascontiguousarray(field[::-1, :])


def read_fields(path: str, params: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with open(path, "rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                break
            short = eccodes.codes_get(gid, "shortName")
            if short in params and short not in out:
                out[short] = (
                    eccodes.codes_get_values(gid).reshape(SHAPE).astype(np.float32)
                )
            eccodes.codes_release(gid)
    missing = [p for p in params if p not in out]
    if missing:
        raise RuntimeError(f"GRIB is missing {missing}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-04-30")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--source", default="azure")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        grib = os.path.join(tmp, "static.grib2")
        Client(source=args.source, model="aifs-single").retrieve(
            target=grib,
            date=args.date,
            time=0,
            stream="oper",
            type="fc",
            levtype="sfc",
            param=["lsm", "z", "sdor"],
            step=0,
        )
        fields = read_fields(grib, ["lsm", "z", "sdor"])

    land = to_zeus_grid(np.clip(fields["lsm"], 0.0, 1.0))
    orography_km = to_zeus_grid(fields["z"]) / np.float32(GRAVITY * 1000.0)
    # Subgrid orography standard deviation, scaled to a bounded range.
    roughness_km = to_zeus_grid(np.clip(fields["sdor"], 0.0, None)) / np.float32(1000.0)

    np.save(os.path.join(args.output_dir, "land_sea.npy"), land)
    np.save(
        os.path.join(args.output_dir, "orography.npy"),
        orography_km.astype(np.float32, copy=False),
    )
    np.save(
        os.path.join(args.output_dir, "roughness.npy"),
        roughness_km.astype(np.float32, copy=False),
    )
    print(
        {
            "output": args.output_dir,
            "land_mean": round(float(land.mean()), 4),
            "orography_km_min": round(float(orography_km.min()), 3),
            "orography_km_max": round(float(orography_km.max()), 3),
            "roughness_km_max": round(float(roughness_km.max()), 4),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
