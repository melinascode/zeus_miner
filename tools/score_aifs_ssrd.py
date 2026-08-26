"""Score AIFS Single SSRD reconstructions vs ERA5 (48h / 360h iwRMSE/iwMAE).

AIFS publishes accumulated J/m² every 6 hours. Two hourly reconstructions:

  constant  — spread each 6h interval as a flat flux
  zenith    — spread the same interval energy with max(cos zenith, 0)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import eccodes
import numpy as np
import torch
import xarray as xr

from zeus.utils.region_mask import geographic_scalar_for_variable
from zeus_ml.datasets.aifs_downscale_dataset import era5_to_zeus_grid, to_zeus_grid
from zeus_ml.evaluate.evaluate_aifs_downscaler import Stream, cycle_maps
from zeus_ml.models.aifs_downscaler_cnn import FULL_HEIGHT, FULL_WIDTH, MAX_LEAD_HOURS, STEP_HOURS
from zeus_ml.models.lead_aware_residual_cnn_v4 import cosine_solar_zenith

N_STEPS = MAX_LEAD_HOURS // STEP_HOURS + 1
SECONDS_PER_HOUR = 3600.0
ZENITH_FLOOR = 1e-6


def decode_accumulated_ssrd(path: Path) -> np.ndarray:
    """Return accumulated J/m² on the Zeus grid, shape (61, 721, 1440)."""

    out = np.zeros((N_STEPS, FULL_HEIGHT, FULL_WIDTH), np.float32)
    seen = np.zeros(N_STEPS, bool)
    with open(path, "rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                break
            step = int(eccodes.codes_get(gid, "endStep"))
            if step % STEP_HOURS == 0 and 0 <= step <= MAX_LEAD_HOURS:
                index = step // STEP_HOURS
                values = eccodes.codes_get_values(gid).reshape(FULL_HEIGHT, FULL_WIDTH)
                out[index] = to_zeus_grid(values)
                seen[index] = True
            eccodes.codes_release(gid)
    if not seen.all():
        missing = np.flatnonzero(~seen).tolist()
        raise ValueError(f"{path.name} missing SSRD steps {missing[:12]}")
    return np.clip(out, 0.0, None)


def _hour_zenith_weight(
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    cycle_time: datetime,
    lead: int,
    samples: int,
) -> torch.Tensor:
    """Mean daytime cosine zenith over the hour ending at `lead`."""

    if lead <= 0:
        return cosine_solar_zenith(latitudes, longitudes, cycle_time).clamp_min(0.0)
    acc = torch.zeros((FULL_HEIGHT, FULL_WIDTH), dtype=torch.float32)
    for i in range(samples):
        offset = lead - 1 + (i + 0.5) / samples
        valid = cycle_time + timedelta(hours=offset)
        acc = acc + cosine_solar_zenith(latitudes, longitudes, valid).clamp_min(0.0)
    return acc / float(samples)


def reconstruct_hourly(
    accumulated: np.ndarray,
    *,
    method: str,
    cycle_time: datetime,
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    zenith_samples: int,
) -> np.ndarray:
    """Hourly mean W/m², shape (361, 721, 1440). Lead 0 is zero."""

    hourly = np.zeros((MAX_LEAD_HOURS + 1, FULL_HEIGHT, FULL_WIDTH), np.float32)
    for index in range(1, N_STEPS):
        start = (index - 1) * STEP_HOURS
        end = index * STEP_HOURS
        energy = np.clip(accumulated[index] - accumulated[index - 1], 0.0, None)
        leads = list(range(start + 1, end + 1))
        if method == "constant":
            flux = (energy / (STEP_HOURS * SECONDS_PER_HOUR)).astype(np.float32)
            for lead in leads:
                hourly[lead] = flux
            continue
        if method != "zenith":
            raise ValueError(f"unknown method {method}")
        weights = [
            _hour_zenith_weight(
                latitudes, longitudes, cycle_time, lead, zenith_samples
            )
            .numpy()
            .astype(np.float32)
            for lead in leads
        ]
        total = np.sum(weights, axis=0)
        safe = np.maximum(total, ZENITH_FLOOR)
        for lead, weight in zip(leads, weights):
            scale = np.where(total > ZENITH_FLOOR, weight / safe, 1.0 / STEP_HOURS)
            hourly[lead] = (energy * scale / SECONDS_PER_HOUR).astype(np.float32)
    return hourly


class Era5SsrdReader:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._open: dict[str, xr.Dataset] = {}

    def read(self, valid_time: datetime) -> np.ndarray:
        day = valid_time.strftime("%Y-%m-%d")
        if day not in self._open:
            path = self.root / "surface_solar_radiation_downwards" / f"era5_{day}.nc"
            self._open[day] = xr.open_dataset(path, engine="h5netcdf")
        array = self._open[day]["ssrd"]
        values = array.isel({array.dims[0]: valid_time.hour}).values
        watts = np.asarray(values, dtype=np.float32) / SECONDS_PER_HOUR
        return era5_to_zeus_grid(np.clip(watts, 0.0, None))


def fmt(row: dict) -> str:
    return f"ssrd={row['rmse']:.3f}/{row['mae']:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", default="20260801T000000Z")
    parser.add_argument(
        "--grib",
        default="/Zeus/data/evaluation/aifs_single/20260801T000000Z.ssrd.grib2",
    )
    parser.add_argument("--era5-root", default="/Zeus/data/evaluation/era5")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--zenith-samples", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    cycle_time = datetime.strptime(args.cycle, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    latitudes = torch.linspace(-90.0, 90.0, FULL_HEIGHT)
    longitudes = torch.arange(-180.0, 180.0, 0.25)
    _, current_weights = cycle_maps(cycle_time, latitudes)
    cosine = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)[:, None]
    solar_metric = cosine * geographic_scalar_for_variable(
        "surface_solar_radiation_downwards"
    )
    weight_maps = {
        "current": current_weights,
        "official_solar": solar_metric / solar_metric.mean(),
    }
    truth = Era5SsrdReader(args.era5_root)

    print(f"decoding {args.grib}", flush=True)
    accumulated = decode_accumulated_ssrd(Path(args.grib))
    methods = ("constant", "zenith")
    forecasts = {}
    for method in methods:
        print(f"  reconstructing {method}", flush=True)
        forecasts[method] = reconstruct_hourly(
            accumulated,
            method=method,
            cycle_time=cycle_time,
            latitudes=latitudes,
            longitudes=longitudes,
            zenith_samples=args.zenith_samples,
        )
        print(
            f"    mean W/m2 {forecasts[method].mean():.2f}  "
            f"max {forecasts[method].max():.1f}",
            flush=True,
        )

    windows = (12, 24, 48, 360)
    streams = {
        method: {
            wname: {h: Stream(1) for h in windows} for wname in weight_maps
        }
        for method in methods
    }
    for lead in range(MAX_LEAD_HOURS + 1):
        truth_hour = torch.from_numpy(
            truth.read(cycle_time + timedelta(hours=lead))
        ).unsqueeze(0)
        for method in methods:
            pred = torch.from_numpy(forecasts[method][lead]).unsqueeze(0)
            for wname, weights in weight_maps.items():
                for horizon, stream in streams[method][wname].items():
                    if lead <= horizon:
                        stream.update(pred, truth_hour, weights)
        if lead in windows or lead % 120 == 0:
            print(f"  lead {lead}", flush=True)

    print(f"{args.cycle} AIFS SSRD")
    for horizon in windows:
        print(f"  {horizon}h")
        for method in methods:
            for wname in weight_maps:
                row = streams[method][wname][horizon].finalize()[0]
                print(f"    {method:10s} {wname:15s} {fmt(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
