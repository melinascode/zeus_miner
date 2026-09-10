#!/usr/bin/env python3
"""Empirical IFS-ENS init-aging penalty at fixed valid time.

The 0727 00z and 0801 00z IFS means overlap on valid times Aug 7 01z .. Aug 11
00z (0801 leads 145..240 vs 0727 leads 265..360). Scoring both against ERA5 at
the same valid hours measures how much a 120h-older init costs at day 6-15
valid times. The lagged borrow for 06/18z cubes ages the init by only 6h.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from zeus.utils.coordinates import get_grid
from zeus.utils.region_mask import geographic_scalar_for_variable, region_masks_for_grid
from zeus.validator.constants import LATITUDE_WEIGHTS_PATH
from zeus_ml.datasets.aifs_downscale_dataset import era5_to_zeus_grid

ERA5 = Path("/Zeus/data/evaluation/era5")
FRESH = np.load("/Zeus/data/evaluation/ifs_ens_mean/20260801T000000Z.npy")
OLD = np.load("/Zeus/data/evaluation/ifs_ens_mean/20260727T000000Z.npy")
T0 = datetime(2026, 8, 1, tzinfo=timezone.utc)
VARS = (("100m_u_component_of_wind", "u100", 1), ("100m_v_component_of_wind", "v100", 2))


def interp(arr: np.ndarray, channel: int, lead: int) -> np.ndarray:
    left, rem = divmod(min(lead, 360), 6)
    f = arr[left, channel].astype(np.float64)
    if rem:
        a = rem / 6.0
        f = (1.0 - a) * f + a * arr[left + 1, channel].astype(np.float64)
    return f


def main() -> int:
    coslat = np.load(LATITUDE_WEIGHTS_PATH).astype(np.float64)[:, None]
    masks = region_masks_for_grid(get_grid(-90.0, 90.0, -180.0, 179.75))
    regions = {"global": 1.0, "germany": masks["germany"].numpy().astype(np.float64)}
    acc = {v: {r: {"fresh": [0.0, 0.0, 0.0], "old": [0.0, 0.0, 0.0], "diff": [0.0, 0.0]} for r in regions} for v, _, _ in VARS}

    open_ds: dict = {}

    def truth(variable, code, valid):
        day = valid.strftime("%Y-%m-%d")
        key = (variable, day)
        if key not in open_ds:
            if len(open_ds) > 4:
                open_ds.pop(next(iter(open_ds))).close()
            open_ds[key] = xr.open_dataset(ERA5 / variable / f"era5_{day}.nc", engine="h5netcdf")
        ds = open_ds[key]
        vals = ds[code].isel({ds[code].dims[0]: valid.hour}).values
        return era5_to_zeus_grid(np.asarray(vals, np.float32)).astype(np.float64)

    for lead in range(145, 241):  # 0801 leads; 0727 lead = lead + 120
        valid = T0 + timedelta(hours=lead)
        for var, code, ch in VARS:
            geo = geographic_scalar_for_variable(var).numpy().astype(np.float64)
            tr = truth(var, code, valid)
            f_new = interp(FRESH, ch, lead)
            f_old = interp(OLD, ch, lead + 120)
            for rname, mask in regions.items():
                w = coslat * geo * (mask if isinstance(mask, np.ndarray) else 1.0)
                m = float(w.sum())
                for tag, f in (("fresh", f_new), ("old", f_old)):
                    e = f - tr
                    acc[var][rname][tag][0] += float((e * e * w).sum())
                    acc[var][rname][tag][1] += float((np.abs(e) * w).sum())
                    acc[var][rname][tag][2] += m
                d = f_new - f_old
                acc[var][rname]["diff"][0] += float((d * d * w).sum())
                acc[var][rname]["diff"][1] += m
        if lead % 24 == 0:
            print(f"lead {lead}/240", flush=True)

    print("\nvalid times Aug 7 01z .. Aug 11 00z; init age fresh=6-10d, old=11-15d (+120h)")
    for var, _, _ in VARS:
        for rname in regions:
            a = acc[var][rname]
            r_new = np.sqrt(a["fresh"][0] / a["fresh"][2]); m_new = a["fresh"][1] / a["fresh"][2]
            r_old = np.sqrt(a["old"][0] / a["old"][2]); m_old = a["old"][1] / a["old"][2]
            d_rms = np.sqrt(a["diff"][0] / a["diff"][1])
            pen = (r_old / r_new - 1.0) * 100.0
            print(f"{var:28s} [{rname:7s}] fresh {r_new:.3f}/{m_new:.3f}  +120h-old {r_old:.3f}/{m_old:.3f}  penalty {pen:+.1f}%  field-diff RMS {d_rms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
