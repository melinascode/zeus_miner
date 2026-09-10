#!/usr/bin/env python3
"""0.01-grid search for live blend knobs (IFS wind ramp, short SSRD, fresh TUV).

Does not touch the miner, builder, or ForecastStore.
"""

from __future__ import annotations

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

ERA5 = Path("/Zeus/data/evaluation/era5")
OUT = Path("/Zeus/data/evaluation/training/tune_alphas_001.json")
ALPHAS = np.round(np.arange(0.0, 1.001, 0.01), 2)
RAMP_START, RAMP_END = 72, 360
LIVE_AMAX = 0.65
N_A = len(ALPHAS)

CODES = {
    "2m_temperature": "t2m",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "surface_solar_radiation_downwards": "ssrd",
}

FRESH_JOBS = (
    ("20260801T00Z_v3", "20260801T000000Z", 3),
    ("20260826T00Z_v3", "20260826T000000Z", 3),
    ("20260826T06Z", "20260826T060000Z", 4),
    ("20260826T12Z_v3", "20260826T120000Z", 3),
    ("20260827T00Z_v3", "20260827T000000Z", 3),
)


def weights_for(variable: str) -> dict[str, np.ndarray]:
    coslat = np.load(LATITUDE_WEIGHTS_PATH).astype(np.float64)[:, None]
    geo = geographic_scalar_for_variable(variable).numpy().astype(np.float64)
    masks = region_masks_for_grid(get_grid(-90.0, 90.0, -180.0, 179.75))
    base = coslat * geo
    return {
        "global": base,
        "europe": base * masks["europe"].numpy(),
        "germany": base * masks["germany"].numpy(),
    }


class Era5:
    def __init__(self) -> None:
        self._open: dict = {}

    def read(self, variable: str, valid: datetime, solar: bool) -> np.ndarray:
        day = valid.strftime("%Y-%m-%d")
        key = (variable, day)
        if key not in self._open:
            if len(self._open) > 6:
                self._open.pop(next(iter(self._open))).close()
            self._open[key] = xr.open_dataset(
                ERA5 / variable / f"era5_{day}.nc", engine="h5netcdf"
            )
        ds = self._open[key]
        code = CODES[variable]
        values = ds[code].isel({ds[code].dims[0]: valid.hour}).values
        field = era5_to_zeus_grid(np.asarray(values, np.float32)).astype(np.float64)
        if solar:
            field = np.clip(field / 3600.0, 0.0, None)
        return field

    def close(self) -> None:
        for ds in self._open.values():
            ds.close()
        self._open.clear()


def ifs_interp(ifs: np.ndarray, channel: int, lead: int) -> np.ndarray:
    lead = min(int(lead), 360)
    left, rem = divmod(lead, 6)
    field = ifs[left, channel].astype(np.float64)
    if rem:
        frac = rem / 6.0
        field = (1.0 - frac) * field + frac * ifs[left + 1, channel].astype(np.float64)
    return field


def ramp_factor(lead: int) -> float:
    if lead <= RAMP_START:
        return 0.0
    return min(1.0, (lead - RAMP_START) / float(RAMP_END - RAMP_START))


def pack(sq: np.ndarray, ab: np.ndarray, mass: float) -> list[dict]:
    rmse = np.sqrt(sq / mass)
    mae = ab / mass
    out = []
    for i, a in enumerate(ALPHAS):
        out.append(
            {
                "alpha": float(a),
                "rmse": float(rmse[i]),
                "mae": float(mae[i]),
                "combined": float((rmse[i] + mae[i]) / 2.0),
            }
        )
    return out


def argmin_combined(rows: list[dict]) -> dict:
    return min(rows, key=lambda r: r["combined"])


def at_alpha(rows: list[dict], target: float) -> dict:
    return min(rows, key=lambda r: abs(r["alpha"] - target))


def accumulate_blend(
    err0: np.ndarray, delta: np.ndarray, w: np.ndarray, sq: np.ndarray, ab: np.ndarray
) -> None:
    """err = err0 + alpha * delta. Updates sq/ab in place for every ALPHAS."""
    e0 = err0
    d = delta
    se2 = float((e0 * e0 * w).sum())
    sed = float((e0 * d * w).sum())
    sd2 = float((d * d * w).sum())
    sq += se2 + 2.0 * ALPHAS * sed + (ALPHAS * ALPHAS) * sd2
    for i in range(0, N_A, 10):
        sl = ALPHAS[i : i + 10]
        # (k, H, W) abs-error, k<=10
        err = e0[None] + sl[:, None, None] * d[None]
        ab[i : i + 10] += (np.abs(err) * w[None]).sum(axis=(1, 2))


def tune_wind(era5: Era5) -> dict:
    print("=== IFS wind ramp alpha_max (0.01) ===", flush=True)
    cycle = "20260801T000000Z"
    cycle_time = datetime.strptime(cycle, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    cube = np.load(
        "/Zeus/data/evaluation/scoring_cubes/20260801T000000Z_v3/hourly_tuv_f16.npy",
        mmap_mode="r",
    )
    ifs = np.load("/Zeus/data/evaluation/ifs_ens_mean/20260801T000000Z.npy")
    wmap = {
        "100m_u_component_of_wind": {"global": weights_for("100m_u_component_of_wind")["global"]},
        "100m_v_component_of_wind": {"global": weights_for("100m_v_component_of_wind")["global"]},
    }
    windows = {"h48": (0, 48), "h360": (0, 360), "tail73": (73, 360)}
    acc = {
        var: {
            "global": {win: [np.zeros(N_A), np.zeros(N_A), 0.0] for win in windows}
        }
        for var in wmap
    }
    for lead in range(361):
        rf = ramp_factor(lead)
        a_live = LIVE_AMAX * rf
        valid = cycle_time + timedelta(hours=lead)
        for var, ch in (("100m_u_component_of_wind", 1), ("100m_v_component_of_wind", 2)):
            truth = era5.read(var, valid, False)
            blended = np.asarray(cube[lead, ch], np.float64)
            native = ifs_interp(ifs, ch, lead)
            cnn = blended if a_live < 1e-9 else (blended - a_live * native) / (1.0 - a_live)
            err0 = cnn - truth
            delta = (native - cnn) * rf  # extra error when amax = 1
            for reg, w in wmap[var].items():
                mass = float(w.sum())
                inc_sq = np.zeros(N_A)
                inc_ab = np.zeros(N_A)
                if rf == 0.0:
                    se2 = float((err0 * err0 * w).sum())
                    se1 = float((np.abs(err0) * w).sum())
                    inc_sq[:] = se2
                    inc_ab[:] = se1
                else:
                    accumulate_blend(err0, delta, w, inc_sq, inc_ab)
                for win, (lo, hi) in windows.items():
                    if lo <= lead <= hi:
                        acc[var][reg][win][0] += inc_sq
                        acc[var][reg][win][1] += inc_ab
                        acc[var][reg][win][2] += mass
        if lead % 48 == 0:
            print(f"  wind lead {lead}/360", flush=True)

    result = {"cycle": cycle, "live_amax": LIVE_AMAX, "curves": {}}
    summary = {}
    for var in wmap:
        result["curves"][var] = {"global": {}}
        summary[var] = {"global": {}}
        for win in windows:
            sq, ab, mass = acc[var]["global"][win]
            rows = pack(sq, ab, mass)
            result["curves"][var]["global"][win] = rows
            best = argmin_combined(rows)
            live = at_alpha(rows, LIVE_AMAX)
            summary[var]["global"][win] = {
                "best": best,
                "live_0.65": live,
                "cnn_only": at_alpha(rows, 0.0),
            }
            print(
                f"  {var} global {win}: best amax={best['alpha']:.2f} "
                f"C={best['combined']:.4f}  live0.65 C={live['combined']:.4f} "
                f"cnn {at_alpha(rows, 0.0)['combined']:.4f}",
                flush=True,
            )
    result["summary"] = summary
    return result


def tune_fresh(era5: Era5) -> dict:
    print("=== fresh/stale TUV alpha (0.01) ===", flush=True)
    jobs = {}
    for folder, cycle, nch in FRESH_JOBS:
        root = Path("/Zeus/data/evaluation/freshness_cubes") / folder
        fresh = np.load(root / ("fresh_cnn_v3.npy" if (root / "fresh_cnn_v3.npy").is_file() else "fresh_cnn.npy"), mmap_mode="r")
        stale = np.load(root / ("stale_serving_v3.npy" if (root / "stale_serving_v3.npy").is_file() else "stale_serving.npy"), mmap_mode="r")
        n = min(fresh.shape[0], stale.shape[0], 49)
        cycle_time = datetime.strptime(cycle, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        print(f"  {folder} n={n} ch={nch}", flush=True)
        var_acc = {}
        for var, ch in (
            ("2m_temperature", 0),
            ("100m_u_component_of_wind", 1),
            ("100m_v_component_of_wind", 2),
        ):
            wsets = {"global": weights_for(var)["global"]}
            acc = {reg: [np.zeros(N_A), np.zeros(N_A), 0.0] for reg in wsets}
            for hour in range(n):
                valid = cycle_time + timedelta(hours=hour)
                truth = era5.read(var, valid, False)
                f = np.asarray(fresh[hour, ch], np.float64)
                s = np.asarray(stale[hour, ch], np.float64)
                err0 = s - truth  # alpha=0 -> stale
                delta = f - s  # extra when alpha=1 (all fresh)
                for reg, w in wsets.items():
                    acc[reg][2] += float(w.sum())
                    accumulate_blend(err0, delta, w, acc[reg][0], acc[reg][1])
            var_acc[var] = {
                reg: pack(acc[reg][0], acc[reg][1], acc[reg][2]) for reg in wsets
            }
        jobs[cycle] = var_acc

    live_a = {
        "2m_temperature": 0.45,
        "100m_u_component_of_wind": 0.65,
        "100m_v_component_of_wind": 0.60,
    }
    pooled = {var: [np.zeros(N_A), np.zeros(N_A)] for var in live_a}
    for cycle, var_acc in jobs.items():
        for var, regs in var_acc.items():
            rows = regs["global"]
            for i, row in enumerate(rows):
                pooled[var][0][i] += row["rmse"]
                pooled[var][1][i] += row["mae"]
    n_cyc = len(jobs)
    pooled_rows = {}
    for var in live_a:
        rows = []
        for i, a in enumerate(ALPHAS):
            rmse = float(pooled[var][0][i] / n_cyc)
            mae = float(pooled[var][1][i] / n_cyc)
            rows.append({"alpha": float(a), "rmse": rmse, "mae": mae, "combined": (rmse + mae) / 2.0})
        pooled_rows[var] = rows
        best = argmin_combined(rows)
        live = at_alpha(rows, live_a[var])
        print(
            f"  POOLED {var}: best a={best['alpha']:.2f} C={best['combined']:.4f}  "
            f"live {live_a[var]:.2f} C={live['combined']:.4f}",
            flush=True,
        )
    return {"cycles": {k: {v: r["global"] for v, r in val.items()} for k, val in jobs.items()}, "pooled": pooled_rows, "live": live_a}


def tune_ssrd(era5: Era5) -> dict:
    """Blend cube 6h-zenith vs IFS-ENS 6h-zenith (proxy for live 0.4/0.6 vs 3h)."""
    print("=== short SSRD stale vs IFS hourly (0.01); 3h IFS not on disk ===", flush=True)
    from tools.score_aifs_ssrd import reconstruct_hourly

    cycle = "20260801T000000Z"
    cycle_time = datetime.strptime(cycle, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    cube = np.load(
        "/Zeus/data/evaluation/scoring_cubes/20260801T000000Z_v3/hourly_ssrd_f16.npy",
        mmap_mode="r",
    )
    acc_ifs = np.load("/Zeus/data/evaluation/ifs_ens_ssrd_mean/20260801T000000Z.npy")
    lats = torch.linspace(-90.0, 90.0, 721)
    lons = torch.arange(-180.0, 180.0, 0.25)
    print("  reconstructing IFS 6h SSRD 0..60h zenith ...", flush=True)
    ifs_h = reconstruct_hourly(
        acc_ifs[:11].astype(np.float32),
        method="zenith",
        cycle_time=cycle_time,
        latitudes=lats,
        longitudes=lons,
        zenith_samples=4,
        step_hours=6,
    )
    wsets = {"global": weights_for("surface_solar_radiation_downwards")["global"]}
    windows = {
        "00z_0_48": (0, 0, 49),  # challenge=00z, cube hours 0..48
        "12z_12_60": (12, 12, 49),  # live-like: challenge=12z, cube 12..60
    }
    out = {}
    var = "surface_solar_radiation_downwards"
    for name, (challenge_hour, cube_off, n) in windows.items():
        ch_time = cycle_time + timedelta(hours=challenge_hour)
        acc = {reg: [np.zeros(N_A), np.zeros(N_A), 0.0] for reg in wsets}
        for hour in range(n):
            valid = ch_time + timedelta(hours=hour)
            truth = era5.read(var, valid, True)
            stale = np.asarray(cube[cube_off + hour], np.float64)
            ifs = np.asarray(ifs_h[cube_off + hour], np.float64)
            err0 = stale - truth  # alpha=0 -> all stale 6h
            delta = ifs - stale  # alpha=1 -> all IFS
            for reg, w in wsets.items():
                acc[reg][2] += float(w.sum())
                accumulate_blend(err0, delta, w, acc[reg][0], acc[reg][1])
        out[name] = {reg: pack(acc[reg][0], acc[reg][1], acc[reg][2]) for reg in wsets}
        best = argmin_combined(out[name]["global"])
        live = at_alpha(out[name]["global"], 0.60)  # live is 0.6 * IFS3h
        print(
            f"  {name} global: best IFS-weight={best['alpha']:.2f} C={best['combined']:.4f}  "
            f"live0.60 C={live['combined']:.4f}  stale0 C={at_alpha(out[name]['global'], 0.0)['combined']:.4f}",
            flush=True,
        )
    return {"note": "IFS member is 6h-zenith (no 3h archive for 0801); live 0.6 is vs 3h IFS", "cycle": cycle, "windows": out}


def main() -> int:
    era5 = Era5()
    payload = {"alphas": [float(a) for a in ALPHAS], "wind": None, "fresh": None, "ssrd": None}
    try:
        payload["wind"] = tune_wind(era5)
        payload["fresh"] = tune_fresh(era5)
        payload["ssrd"] = tune_ssrd(era5)
    finally:
        era5.close()
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
