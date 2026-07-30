from __future__ import annotations

import numpy as np
import torch


def persistence_from_initial_field(
    forecast: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Repeat candidate H000 without using future ground truth."""

    if isinstance(forecast, torch.Tensor):
        tensor = forecast.detach().to(device="cpu")
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(forecast))
    if tensor.ndim != 3 or tensor.shape[0] < 1:
        raise ValueError(
            "Forecast must have shape (time, latitude, longitude) with "
            "at least one time step."
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("Forecast contains NaN or Inf.")
    return tensor[0:1].expand_as(tensor)
