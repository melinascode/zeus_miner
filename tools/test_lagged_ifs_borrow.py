#!/usr/bin/env python3
"""Test the lagged-IFS borrow for 06/18z cubes on the 0801 00z replay.

Production context: 06/18z IFS-ENS stops at 144h, so cube leads 145..360 of
those runs carry no IFS wind blend. Proposal: fill leads 145..354 with the
previous 00/12z run's IFS mean at lead L+6 (same valid time, 6h older init)
and hold IFS(360) for 355..360, keeping the same alpha(L) ramp.

Simulation on the 20260801T00z V3 scoring cube (which has the native ramp
baked in; we un-blend exactly since alpha is known):

  cnn_only       un-blended V3 CNN winds (what 06/18z serves past 144h)
  current_0618   native ramp to 144, CNN after        <- today's 06/18z
  proposed_0618  native to 144, then IFS(L+6) borrow, IFS(360) hold at 355+
  current_0012   full native ramp                     <- ceiling (00/12z)
  ifs_only       raw IFS mean interp (skill-decay curve -> lag penalty)

Caveat: with a single run, the "borrowed" IFS(L+6) field verifies 6h after
the truth hour, so proposed_0618 is a PESSIMISTIC proxy. In production the
borrowed field verifies at the correct valid time (init is 6h older instead).

Scores: official wind scalars x cos-lat, iwRMSE/iwMAE, global + Europe box +
Germany box, segments 73-144 / 145-354 / 355-360 / 145-360.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from zeus.utils.coordinates import get_grid
from zeus.utils.region_mask import geographic_scalar_for_variable, region_masks_for_grid
from zeus.validator.constants import LATITUDE_WEIGHTS_PATH
from zeus_ml.datasets.aifs_downscale_dataset import era5_to_zeus_grid

CYCLE = "20260801T000000Z"
CUBE = Path("/Zeus/data/evaluation/scoring_cubes/20260801T000000Z_v3/hourly_tuv_f16.npy")
IFS = Path("/Zeus/data/evaluation/ifs_ens_mean/20260801T000000Z.npy")
ERA5 = Path("/Zeus/data/evaluation/era5")
OUT = Path("/Zeus/data/evaluation/training/lagged_ifs_borrow_0801.json")

RAMP_START, RAMP_END, ALPHA_MAX = 72, 360, 0.65
VARS = (("100m_u_component_of_wind", 1), ("100m_v_component_of_wind", 2))
VARIANTS = ("cnn_only", "current_0618", "proposed_0618", "current_0012", "ifs_only")
SEGMENTS = {"73_144": (73, 144), "145_354": (145, 354), "355_360": (355, 360), "145_360": (145, 360)}


def alpha(lead: int) -> float:
    if lead <= RAMP_START:
        return 0.0
    return ALPHA_MAX * min(1.0, (lead - RAMP_START) / float(RAMP_END - RAMP_START))


def ifs_interp(ifs: np.ndarray, channel: int, lead: int) -> np.ndarray:
    """Linear interp of 6h steps, matching apply_ifs_wind_ramp."""
    lead = min(lead, 360)
    left, rem = divmod(lead, 6)
    field = ifs[left, channel].astype(np.float32)
    if rem:
        frac = rem / 6.0
        field = (1.0 - frac) * field + frac * ifs[left + 1, channel].astype(np.float32)
    return field


def main() -> int:
    cycle_time = datetime.strptime(CYCLE, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    cube = np.load(CUBE, mmap_mode="r")
    ifs = np.load(IFS)  # (61, 3, 721, 1440) fp16, channels t2m/u/v

    coslat = np.load(LATITUDE_WEIGHTS_PATH).astype(np.float64)[:, None]
    masks = region_masks_for_grid(get_grid(-90.0, 90.0, -180.0, 179.75))
    regions = {
        "global": np.ones((721, 1440), np.float64),
        "europe": masks["europe"].numpy().astype(np.float64),
        "germany": masks["germany"].numpy().astype(np.float64),
    }
    weights = {}
    for var, _ in VARS:
        geo = geographic_scalar_for_variable(var).numpy().astype(np.float64)
        weights[var] = {name: coslat * geo * mask for name, mask in regions.items()}

    # accumulators: [variant][var][region][segment] -> [sq_sum, abs_sum, mass]
    acc = {
        v: {var: {r: {s: [0.0, 0.0, 0.0] for s in SEGMENTS} for r in regions} for var, _ in VARS}
        for v in VARIANTS
    }
    ifs_curve = {var: {} for var, _ in VARS}  # 6h-step skill for lag penalty

    open_ds: dict = {}

    def truth_hour(variable: str, code: str, valid: datetime) -> np.ndarray:
        day = valid.strftime("%Y-%m-%d")
        key = (variable, day)
        if key not in open_ds:
            if len(open_ds) > 4:
                open_ds.pop(next(iter(open_ds)))[1].close()
            ds = xr.open_dataset(ERA5 / variable / f"era5_{day}.nc", engine="h5netcdf")
            open_ds[key] = (variable, ds)
        ds = open_ds[key][1]
        values = ds[code].isel({ds[code].dims[0]: valid.hour}).values
        return era5_to_zeus_grid(np.asarray(values, dtype=np.float32)).astype(np.float64)

    codes = {"100m_u_component_of_wind": "u100", "100m_v_component_of_wind": "v100"}

    for lead in range(73, 361):
        a = alpha(lead)
        valid = cycle_time + timedelta(hours=lead)
        for var, channel in VARS:
            truth = truth_hour(var, codes[var], valid)
            blended = np.asarray(cube[lead, channel], np.float64)  # native ramp baked in
            native_ifs = ifs_interp(ifs, channel, lead).astype(np.float64)
            cnn = (blended - a * native_ifs) / (1.0 - a)  # exact un-blend

            fields = {
                "cnn_only": cnn,
                "current_0012": blended,
                "ifs_only": native_ifs,
            }
            fields["current_0618"] = blended if lead <= 144 else cnn
            if lead <= 144:
                prop = blended
            elif lead <= 354:
                prop = (1.0 - a) * cnn + a * ifs_interp(ifs, channel, lead + 6).astype(np.float64)
            else:
                prop = (1.0 - a) * cnn + a * ifs[60, channel].astype(np.float64)  # hold IFS(360)
            fields["proposed_0618"] = prop

            for vname, pred in fields.items():
                err = pred - truth
                for rname, w in weights[var].items():
                    sq = float((err * err * w).sum())
                    ab = float((np.abs(err) * w).sum())
                    m = float(w.sum())
                    for sname, (lo, hi) in SEGMENTS.items():
                        if lo <= lead <= hi:
                            slot = acc[vname][var][rname][sname]
                            slot[0] += sq
                            slot[1] += ab
                            slot[2] += m

            if lead % 6 == 0 and lead >= 144:  # IFS skill decay curve, global
                err = native_ifs - truth
                w = weights[var]["global"]
                ifs_curve[var][lead] = {
                    "rmse": float(np.sqrt((err * err * w).sum() / w.sum())),
                    "mae": float((np.abs(err) * w).sum() / w.sum()),
                }
        if lead % 24 == 0:
            print(f"lead {lead}/360", flush=True)

    result = {"cycle": CYCLE, "note": "proposed_0618 borrow uses 6h valid-time-misaligned IFS (pessimistic proxy)", "scores": {}, "ifs_decay": ifs_curve}
    for vname in VARIANTS:
        result["scores"][vname] = {}
        for var, _ in VARS:
            result["scores"][vname][var] = {}
            for rname in regions:
                result["scores"][vname][var][rname] = {}
                for sname in SEGMENTS:
                    sq, ab, m = acc[vname][var][rname][sname]
                    result["scores"][vname][var][rname][sname] = {
                        "rmse": float(np.sqrt(sq / m)),
                        "mae": float(ab / m),
                    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for var, _ in VARS:
        print(f"\n=== {var} (iwRMSE/iwMAE) ===")
        for rname in regions:
            print(f"  [{rname}]")
            for sname in ("145_354", "355_360", "145_360"):
                row = "    " + sname.replace("_", "-") + ":  "
                for vname in ("cnn_only", "current_0618", "proposed_0618", "current_0012"):
                    s = result["scores"][vname][var][rname][sname]
                    row += f"{vname}={s['rmse']:.3f}/{s['mae']:.3f}  "
                print(row)
    print("\nIFS decay (global u100):")
    for lead in sorted(ifs_curve["100m_u_component_of_wind"]):
        s = ifs_curve["100m_u_component_of_wind"][lead]
        print(f"  lead {lead}: {s['rmse']:.3f}/{s['mae']:.3f}")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
