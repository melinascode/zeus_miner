#!/usr/bin/env python3
"""Stage 1a: extract the Europe training domain into compact fp16 crops.

Domain: 28..79.75N, 40W..51.75E (208 x 368 cells) — the official Europe
scoring box (35-72N, 25W-45E) plus Atlantic upstream context for the CNN.

Outputs under --output-root:
  single/{cycle}.npy   (61, 3, 208, 368) fp16   AIFS Single 6h steps
  ens/{cycle}.npy      (61, 3, 208, 368) fp16   ENS-mean 6h steps
  era5/{day}.npy       (24, 3, 208, 368) fp16   hourly truth
  static.npz           land/orography/roughness/scalars/climatology coefficients
  manifest.json        domain indices + inventory
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zeus.utils.region_mask import geographic_scalar_for_variable
from zeus_ml.datasets.aifs_downscale_dataset import (
    AifsCycleReader,
    EnsMeanCycleReader,
    Era5HourlyReader,
)
from zeus_ml.models.aifs_downscaler_cnn import load_static_maps

LAT_START, LAT_END = 472, 680  # 28.0N .. 79.75N
LON_START, LON_END = 560, 928  # 40.0W .. 51.75E
CROP_HEIGHT = LAT_END - LAT_START
CROP_WIDTH = LON_END - LON_START


def crop(field: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(field, torch.Tensor):
        field = field.numpy()
    return np.ascontiguousarray(
        field[..., LAT_START:LAT_END, LON_START:LON_END]
    )


def convert_cycles(reader, keys: list[str], out_dir: Path, label: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    t0 = time.time()
    for i, key in enumerate(keys):
        target = out_dir / f"{key}.npy"
        if target.is_file():
            done.append(key)
            continue
        try:
            data = reader.get(key)
        except Exception as error:  # corrupt grib should not kill the run
            print(f"  {label} {key} FAILED: {error}", flush=True)
            continue
        np.save(target, crop(data).astype(np.float16))
        done.append(key)
        if i % 25 == 0:
            print(
                f"  {label} {i + 1}/{len(keys)} {key}  {time.time() - t0:.0f}s",
                flush=True,
            )
    print(f"{label}: {len(done)} cycles ready ({time.time() - t0:.0f}s)", flush=True)
    return done


def convert_era5(root: str, out_dir: Path, days: list[str]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = Era5HourlyReader(root, cache_size=2)
    done = []
    t0 = time.time()
    for i, day in enumerate(days):
        target = out_dir / f"{day}.npy"
        if target.is_file():
            done.append(day)
            continue
        base = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        try:
            hours = np.stack(
                [crop(reader.read(base + timedelta(hours=h))) for h in range(24)]
            )
        except Exception as error:
            print(f"  era5 {day} FAILED: {error}", flush=True)
            continue
        np.save(target, hours.astype(np.float16))
        done.append(day)
        if i % 25 == 0:
            print(
                f"  era5 {i + 1}/{len(days)} {day}  {time.time() - t0:.0f}s",
                flush=True,
            )
    print(f"era5: {len(done)} days ready ({time.time() - t0:.0f}s)", flush=True)
    return done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aifs-root", default="/Zeus/data/evaluation/aifs_single")
    parser.add_argument("--ens-root", default="/Zeus/data/evaluation/aifs_ens_mean")
    parser.add_argument("--era5-root", default="/Zeus/data/evaluation/era5")
    parser.add_argument(
        "--static-root", default="/Zeus/data/evaluation/training/aifs_static"
    )
    parser.add_argument(
        "--climatology",
        default="/Zeus/data/evaluation/training/era5_climatology_harmonics.npz",
    )
    parser.add_argument(
        "--output-root", default="/Zeus/data/evaluation/europe_crops"
    )
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    # Static bundle: model inputs + loss weights + climatology, all cropped.
    land, orography, roughness = load_static_maps(args.static_root)
    clim = np.load(args.climatology, allow_pickle=True)
    temp_scalar = geographic_scalar_for_variable("2m_temperature")
    wind_scalar = geographic_scalar_for_variable("100m_u_component_of_wind")
    latitudes = np.linspace(-90.0, 90.0, 721)[LAT_START:LAT_END]
    cosine = np.cos(np.deg2rad(latitudes)).clip(min=0.0)[:, None].astype(np.float32)
    np.savez_compressed(
        out / "static.npz",
        land=crop(land).astype(np.float32),
        orography=crop(orography).astype(np.float32),
        roughness=crop(roughness).astype(np.float32),
        temp_scalar=crop(temp_scalar).astype(np.float32),
        wind_scalar=crop(wind_scalar).astype(np.float32),
        cosine_latitude=np.broadcast_to(
            cosine, (CROP_HEIGHT, CROP_WIDTH)
        ).copy(),
        climatology_coefficients=crop(clim["coefficients"]).astype(np.float32),
        latitudes=latitudes.astype(np.float32),
        longitudes=np.arange(-180.0, 180.0, 0.25)[LON_START:LON_END].astype(
            np.float32
        ),
    )
    print("wrote static.npz", flush=True)

    single_keys = sorted(
        p.stem for p in Path(args.aifs_root).glob("*.grib2")
    )
    ens_keys = sorted(p.stem for p in Path(args.ens_root).glob("*.npy"))
    era5_days = sorted(
        p.stem.replace("era5_", "")
        for p in (Path(args.era5_root) / "2m_temperature").glob("era5_*.nc")
    )

    era5_done = convert_era5(args.era5_root, out / "era5", era5_days)
    ens_done = convert_cycles(
        EnsMeanCycleReader(args.ens_root, cache_size=1), ens_keys, out / "ens", "ens"
    )
    single_done = convert_cycles(
        AifsCycleReader(args.aifs_root, cache_size=1),
        single_keys,
        out / "single",
        "single",
    )

    manifest = {
        "lat_index": [LAT_START, LAT_END],
        "lon_index": [LON_START, LON_END],
        "lat_degrees": [-90.0 + LAT_START * 0.25, -90.0 + (LAT_END - 1) * 0.25],
        "lon_degrees": [-180.0 + LON_START * 0.25, -180.0 + (LON_END - 1) * 0.25],
        "shape": [CROP_HEIGHT, CROP_WIDTH],
        "single_cycles": single_done,
        "ens_cycles": ens_done,
        "era5_days": era5_done,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    print(
        f"manifest: {len(single_done)} single, {len(ens_done)} ens, "
        f"{len(era5_done)} era5 days -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
