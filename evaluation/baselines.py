from __future__ import annotations

import numpy as np
import torch


def persistence_from_initial_field(
    forecast: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Repeat the target-time field from the production-selected GFS cycle.

    The input must already be the Zeus-aligned raw GFS artifact for the
    historical production source-selection (newest ready cycle among offsets
    6/12/18/24), lead mapping, and interpolation. Persistence repeats that
    artifact's H000 (Zeus lead 0 / target-time field) across the horizon.

    Never use ERA5 H000 or any post-issue observation as the persistence
    initial field.
    """

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
