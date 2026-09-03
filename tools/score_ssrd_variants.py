"""Score SSRD source variants on one cycle (official capacity scalars).

Every variant is an accumulated-ssrd array (61, 721, 1440) J/m2 at steps
0..360 by 6. Each is de-accumulated and zenith-redistributed to hourly with
the exact serving code, then scored vs ERA5 at 48h / 360h.

Variants:
  single       AIFS Single (current serving source)
  aifs_ens     AIFS-ENS 51-member mean
  ifs_ens      IFS-ENS 50-member mean
  ens_mix50    (aifs_ens + ifs_ens) / 2
  sgl_ae_50    0.5 single + 0.5 aifs_ens
  ramp_mix     single through 24h, linear to ens_mix50 by 120h
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from evaluation.scoring import ValidatorFaithfulScorer
from zeus.utils.region_mask import geographic_scalar_for_variable
from zeus_ml.datasets.aifs_downscale_dataset import era5_to_zeus_grid
from tools.score_aifs_ssrd import decode_accumulated_ssrd, reconstruct_hourly

HORIZONS = (48, 360)
VARIABLE = "surface_solar_radiation_downwards"


def load_truth(cycle_time: datetime, era5: Path) -> np.ndarray:
    out = np.empty((361, 721, 1440), np.float32)
    open_ds: dict = {}
    for lead in range(361):
        valid = cycle_time + timedelta(hours=lead)
        day = valid.strftime("%Y-%m-%d")
        if day not in open_ds:
            open_ds[day] = xr.open_dataset(
                era5 / VARIABLE / f"era5_{day}.nc", engine="h5netcdf"
            )
            if len(open_ds) > 4:
                open_ds.pop(next(iter(open_ds))).close()
        ds = open_ds[day]
        values = ds["ssrd"].isel({ds["ssrd"].dims[0]: valid.hour}).values
        out[lead] = np.clip(
            era5_to_zeus_grid(np.asarray(values, dtype=np.float32)) / 3600.0, 0.0, None
        )
    for ds in open_ds.values():
        ds.close()
    return out


def ramp_mix(single: np.ndarray, mix: np.ndarray) -> np.ndarray:
    """Accumulation-level ramp is ill-defined; ramp on interval means instead.

    Handled by blending the accumulations with a per-step weight applied to
    the INCREMENTS: rebuild accumulation from blended interval increments.
    """
    inc_s = np.diff(single, axis=0)
    inc_m = np.diff(mix, axis=0)
    steps = np.arange(6, 361, 6, dtype=np.float64)
    alpha = np.clip((steps - 24.0) / (120.0 - 24.0), 0.0, 1.0).astype(np.float32)
    inc = (1.0 - alpha[:, None, None]) * inc_s + alpha[:, None, None] * inc_m
    out = np.zeros_like(single)
    np.cumsum(inc, axis=0, out=out[1:])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--era5-root", default="/Zeus/data/evaluation/era5")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    cycle = args.cycle
    cycle_time = datetime.strptime(cycle, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)

    single = decode_accumulated_ssrd(
        Path(f"/Zeus/data/evaluation/aifs_ssrd/{cycle}.grib2")
    )
    aifs_ens = np.load(f"/Zeus/data/evaluation/aifs_ens_ssrd_mean/{cycle}.npy")
    ifs_ens = np.load(f"/Zeus/data/evaluation/ifs_ens_ssrd_mean/{cycle}.npy")
    ens_mix = 0.5 * aifs_ens + 0.5 * ifs_ens
    variants = {
        "single": single,
        "aifs_ens": aifs_ens,
        "ifs_ens": ifs_ens,
        "ens_mix50": ens_mix,
        "sgl_ae_50": 0.5 * single + 0.5 * aifs_ens,
        "ramp_mix": ramp_mix(single, ens_mix),
    }

    print("loading ERA5 truth ...", flush=True)
    truth = load_truth(cycle_time, Path(args.era5_root))

    scorer = ValidatorFaithfulScorer()
    lat = scorer._latitude_weights(None)
    geo = geographic_scalar_for_variable(VARIABLE)
    weights = lat.view(-1, 1) * geo
    weights = (weights / weights.mean()).numpy()

    latitudes = torch.linspace(-90.0, 90.0, 721)
    longitudes = torch.arange(-180.0, 180.0, 0.25)

    print(f"\n{cycle}  SSRD variants (official scalars)", flush=True)
    print(f"{'variant':12s} {'H':>4} {'iwRMSE':>10} {'iwMAE':>10} {'combined':>10}",
          flush=True)
    for name, accumulated in variants.items():
        hourly = reconstruct_hourly(
            np.clip(accumulated, 0.0, None).astype(np.float32),
            method="zenith",
            cycle_time=cycle_time,
            latitudes=latitudes,
            longitudes=longitudes,
            zenith_samples=4,
        )
        hourly = np.clip(hourly, 0.0, None)
        acc = {h: [0.0, 0.0, 0] for h in HORIZONS}
        for lead in range(361):
            err = hourly[lead] - truth[lead]
            sq = float(np.mean(err * err * weights))
            ab = float(np.mean(np.abs(err) * weights))
            for h in HORIZONS:
                if lead <= h:
                    acc[h][0] += sq
                    acc[h][1] += ab
                    acc[h][2] += 1
        for h in HORIZONS:
            n = acc[h][2]
            rmse = (acc[h][0] / n) ** 0.5
            mae = acc[h][1] / n
            print(f"{name:12s} {h:4d} {rmse:10.3f} {mae:10.3f} {(rmse+mae)/2:10.3f}",
                  flush=True)
        del hourly
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
