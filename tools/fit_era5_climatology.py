#!/usr/bin/env python3
"""Fit per-cell ERA5 climatology: seasonal x diurnal harmonic regression.

For every grid cell and variable (t2m, u100, v100) fit

    y(t) ~ sum_ij  c_ij * A_i(doy) * D_j(hour)

where A = [1, cos/sin annual, cos/sin semi-annual] and D likewise for the
diurnal cycle (25 features). Solved in one streaming pass: accumulate X'X and
X'y over every ERA5 hour, then a single batched solve per variable.

Fit window must end before the held-out test cycles (default 2026-07-11) so
climatology never sees test-period truth.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from zeus_ml.datasets.aifs_downscale_dataset import Era5HourlyReader

N_VARS = 3
HEIGHT, WIDTH = 721, 1440
N_FEATURES = 25


def harmonic_features(when: datetime) -> np.ndarray:
    """25 features: outer product of annual and diurnal harmonic bases."""
    doy = when.timetuple().tm_yday - 1 + when.hour / 24.0
    annual_phase = 2.0 * np.pi * doy / 365.25
    diurnal_phase = 2.0 * np.pi * when.hour / 24.0
    annual = np.array(
        [
            1.0,
            np.cos(annual_phase),
            np.sin(annual_phase),
            np.cos(2 * annual_phase),
            np.sin(2 * annual_phase),
        ]
    )
    diurnal = np.array(
        [
            1.0,
            np.cos(diurnal_phase),
            np.sin(diurnal_phase),
            np.cos(2 * diurnal_phase),
            np.sin(2 * diurnal_phase),
        ]
    )
    return np.outer(annual, diurnal).ravel()


def evaluate_climatology(coefficients: np.ndarray, when: datetime) -> np.ndarray:
    """coefficients (vars, 25, H, W) -> climatology map (vars, H, W)."""
    features = harmonic_features(when).astype(coefficients.dtype)
    return np.tensordot(features, coefficients, axes=([0], [1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--era5-root", default="/Zeus/data/evaluation/era5")
    parser.add_argument("--start", default="2025-04-22")
    parser.add_argument("--end", default="2026-07-11", help="Inclusive last day.")
    parser.add_argument(
        "--output",
        default="/Zeus/data/evaluation/training/era5_climatology_harmonics.npz",
    )
    args = parser.parse_args()

    reader = Era5HourlyReader(args.era5_root, cache_size=2)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    n_cells = HEIGHT * WIDTH
    xtx = np.zeros((N_FEATURES, N_FEATURES), dtype=np.float64)
    xty = np.zeros((N_VARS, N_FEATURES, n_cells), dtype=np.float64)

    n_hours = 0
    n_days_skipped = 0
    day = start
    t0 = time.time()
    while day <= end:
        try:
            for hour in range(24):
                when = day + timedelta(hours=hour)
                values = reader.read(when).reshape(N_VARS, n_cells)
                features = harmonic_features(when)
                xtx += np.outer(features, features)
                xty += features[None, :, None] * values[:, None, :]
                n_hours += 1
        except FileNotFoundError:
            n_days_skipped += 1
        day += timedelta(days=1)
        if day.day in (1, 11, 21):
            print(
                f"  through {day.date()}  hours={n_hours} "
                f"skipped_days={n_days_skipped}  {time.time() - t0:.0f}s",
                flush=True,
            )

    print(f"accumulated {n_hours} hours ({n_days_skipped} days missing)", flush=True)
    # Small ridge keeps the solve stable if harmonics are near-collinear.
    ridge = 1e-6 * np.trace(xtx) / N_FEATURES * np.eye(N_FEATURES)
    coefficients = np.linalg.solve((xtx + ridge)[None], xty).astype(np.float32)
    coefficients = coefficients.reshape(N_VARS, N_FEATURES, HEIGHT, WIDTH)

    residual_check = evaluate_climatology(coefficients, start + timedelta(days=100))
    print(
        "sample climatology means (t2m, u100, v100):",
        [round(float(m), 2) for m in residual_check.mean(axis=(1, 2))],
        flush=True,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        coefficients=coefficients,
        variables=np.array(["t2m", "u100", "v100"]),
        fit_start=args.start,
        fit_end=args.end,
        n_hours=n_hours,
    )
    print(f"wrote {args.output}  ({time.time() - t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
