from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import torch

from evaluation.calibration import FrozenCalibration


SOLAR_VARIABLE = "surface_solar_radiation_downwards"


def apply_calibrated_gfs(
    forecast: torch.Tensor | np.ndarray,
    *,
    variable: str,
    cycle_time: datetime,
    coefficients: FrozenCalibration,
) -> torch.Tensor:
    """Apply frozen additive lead-hour bias; never refit."""

    if isinstance(forecast, torch.Tensor):
        raw = forecast.detach().to(device="cpu", dtype=torch.float32)
    else:
        raw = torch.from_numpy(
            np.ascontiguousarray(forecast, dtype=np.float32)
        )
    if raw.ndim != 3:
        raise ValueError(
            "Forecast must have shape (time, latitude, longitude)."
        )
    if not torch.isfinite(raw).all():
        raise ValueError("Raw GFS forecast contains NaN or Inf.")

    cycle_utc = (
        cycle_time.replace(tzinfo=timezone.utc)
        if cycle_time.tzinfo is None
        else cycle_time.astimezone(timezone.utc)
    )
    bias = coefficients.bias(variable, cycle_utc.hour, raw.shape[0])
    bias_tensor = torch.from_numpy(bias).view(-1, 1, 1)
    calibrated = raw + bias_tensor
    if variable == SOLAR_VARIABLE:
        calibrated = torch.clamp(calibrated, min=0.0)
    if not torch.isfinite(calibrated).all():
        raise ValueError("Calibrated GFS contains NaN or Inf.")
    return calibrated
