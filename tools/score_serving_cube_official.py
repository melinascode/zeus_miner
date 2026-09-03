"""Official capacity-scalar 48h/360h scores of a serving cube (optionally IFS-blended)."""

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
from zeus_ml.serve.live_bundle_builder import apply_ifs_wind_ramp

VARS = (
    ("2m_temperature", "t2m", "tuv", 0, False),
    ("100m_u_component_of_wind", "u100", "tuv", 1, False),
    ("100m_v_component_of_wind", "v100", "tuv", 2, False),
    ("surface_solar_radiation_downwards", "ssrd", "ssrd", None, True),
)
HORIZONS = (48, 360)
LABELS = {
    "2m_temperature": "2m temperature (K)",
    "100m_u_component_of_wind": "100m u-wind (m/s)",
    "100m_v_component_of_wind": "100m v-wind (m/s)",
    "surface_solar_radiation_downwards": "SSRD (W/m²)",
}


class FakeBundle(dict):
    @property
    def files(self):
        return list(self.keys())


def apply_blend(cube: np.ndarray, cycle: str) -> np.ndarray:
    ifs = np.load(f"/Zeus/data/evaluation/ifs_ens_mean/{cycle}.npy").astype(np.float32)
    bundle = FakeBundle(
        ifs_steps=np.arange(0, 361, 6),
        ifs_u100=ifs[:, 1],
        ifs_v100=ifs[:, 2],
    )
    apply_ifs_wind_ramp(cube, bundle, cycle)
    return cube


def score_cube(cycle: str, cube_dir: Path, blended: bool) -> None:
    cycle_time = datetime.strptime(cycle, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    era5 = Path("/Zeus/data/evaluation/era5")
    scorer = ValidatorFaithfulScorer()
    lat = scorer._latitude_weights(None)
    tuv = np.load(cube_dir / "hourly_tuv_f16.npy")
    if blended:
        tuv = apply_blend(tuv.copy(), cycle)
    ssrd = np.load(cube_dir / "hourly_ssrd_f16.npy", mmap_mode="r")
    open_ds: dict = {}

    def truth_hour(variable, code, valid, solar):
        day = valid.strftime("%Y-%m-%d")
        key = (variable, day)
        if key not in open_ds:
            open_ds[key] = xr.open_dataset(
                era5 / variable / f"era5_{day}.nc", engine="h5netcdf"
            )
            if len(open_ds) > 8:
                open_ds.pop(next(iter(open_ds))).close()
        values = open_ds[key][code].isel({open_ds[key][code].dims[0]: valid.hour}).values
        field = era5_to_zeus_grid(np.asarray(values, dtype=np.float32))
        if solar:
            field = np.clip(field / 3600.0, 0.0, None)
        return field

    tag = "ENS-mean + v2 CNN + zenith SSRD + IFS blend" if blended else "serving"
    print(f"\n{cycle}  {tag}", flush=True)
    print(
        f"{'variable':40s} {'H':>4} {'iwRMSE':>10} {'iwMAE':>10} {'combined':>10}",
        flush=True,
    )
    for variable, code, kind, channel, solar in VARS:
        geo = geographic_scalar_for_variable(variable)
        weights = lat.view(-1, 1) * geo
        weights = weights / weights.mean()
        acc = {h: [0.0, 0.0, 0] for h in HORIZONS}
        for lead in range(361):
            valid = cycle_time + timedelta(hours=lead)
            pred = (ssrd[lead] if kind == "ssrd" else tuv[lead, channel]).astype(
                np.float32
            )
            err = torch.from_numpy(pred - truth_hour(variable, code, valid, solar))
            sq = float(err.square().mul(weights).mean())
            ab = float(err.abs().mul(weights).mean())
            for h in HORIZONS:
                if lead <= h:
                    acc[h][0] += sq
                    acc[h][1] += ab
                    acc[h][2] += 1
        for h in HORIZONS:
            n = acc[h][2]
            rmse = (acc[h][0] / n) ** 0.5
            mae = acc[h][1] / n
            print(
                f"{LABELS[variable]:40s} {h:4d} {rmse:10.3f} {mae:10.3f} {(rmse+mae)/2:10.3f}",
                flush=True,
            )
        for ds in open_ds.values():
            ds.close()
        open_ds.clear()


def rebuild_0727(out_dir: Path) -> None:
    """Rebuild the 0727 serving cube (GC'd) then the caller blends it."""
    from tools.score_aifs_ssrd import reconstruct_hourly
    from zeus_ml.datasets.aifs_downscale_dataset import AifsCycleReader, EnsMeanCycleReader
    from zeus_ml.serve.compose_forecast import ForecastComposer

    cycle = "20260727T000000Z"
    cycle_time = datetime.strptime(cycle, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    composer = ForecastComposer()
    ens = EnsMeanCycleReader("/Zeus/data/evaluation/aifs_ens_mean", cache_size=1).get(
        cycle
    )
    single = AifsCycleReader("/Zeus/data/evaluation/aifs_single", cache_size=1).get(
        cycle
    )
    cube = np.empty((361, 3, 721, 1440), dtype=np.float16)
    started = datetime.now(timezone.utc)
    for lead in range(361):
        cube[lead] = composer._apply_global(ens, single, lead, cycle_time).numpy().astype(
            np.float16
        )
        if lead % 48 == 0:
            elapsed = (datetime.now(timezone.utc) - started).total_seconds() / 60.0
            print(f"  rebuild TUV lead {lead}/360 ({elapsed:.1f} min)", flush=True)
    np.save(out_dir / "hourly_tuv_f16.npy", cube)
    del cube, ens, single

    from tools.score_aifs_ssrd import decode_accumulated_ssrd

    accumulated = decode_accumulated_ssrd(
        Path("/Zeus/data/evaluation/aifs_ssrd/20260727T000000Z.grib2")
    )
    hourly = reconstruct_hourly(
        accumulated,
        method="zenith",
        cycle_time=cycle_time,
        latitudes=composer.latitudes,
        longitudes=composer.longitudes,
        zenith_samples=4,
    )
    np.save(out_dir / "hourly_ssrd_f16.npy", np.clip(hourly, 0.0, None).astype(np.float16))
    print(f"wrote {out_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--cube-dir", required=True)
    parser.add_argument("--blend", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    cube_dir = Path(args.cube_dir)
    if args.rebuild:
        rebuild_0727(cube_dir)
    score_cube(args.cycle, cube_dir, blended=args.blend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
