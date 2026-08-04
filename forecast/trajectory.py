from __future__ import annotations

import numpy as np
import xarray as xr

from forecast.base import ForecastModel


class ForecastTrajectoryService:
    """
    Converts a single-lead ForecastModel into a complete forecast trajectory.
    """

    def __init__(self, model: ForecastModel) -> None:
        self.model = model

    def predict(
        self,
        history: xr.DataArray,
        forecast_hours: int,
        include_initial_state: bool = True,
    ) -> xr.DataArray:
        if forecast_hours < 1:
            raise ValueError("forecast_hours must be at least 1.")

        fields: list[xr.DataArray] = []
        lead_times: list[int] = []

        if include_initial_state:
            initial = history.isel(valid_time=-1, drop=True)
            fields.append(initial)
            lead_times.append(0)

        first_lead = 1
        final_lead = forecast_hours - 1 if include_initial_state else forecast_hours

        for lead_hours in range(first_lead, final_lead + 1):
            forecast = self.model.predict(
                history=history,
                lead_hours=lead_hours,
            )

            forecast = forecast.squeeze(drop=True)

            fields.append(forecast)
            lead_times.append(lead_hours)

        trajectory = xr.concat(
            fields,
            dim=xr.IndexVariable("lead_hours", lead_times),
        )

        expected_steps = forecast_hours

        if trajectory.sizes["lead_hours"] != expected_steps:
            raise RuntimeError(
                "Unexpected trajectory length: "
                f"expected {expected_steps}, "
                f"received {trajectory.sizes['lead_hours']}."
            )

        return trajectory

    def to_zeus_array(
        self,
        trajectory: xr.DataArray,
    ) -> np.ndarray:
        required_dimensions = {
            "lead_hours",
            "latitude",
            "longitude",
        }

        missing = required_dimensions.difference(trajectory.dims)

        if missing:
            raise ValueError(
                f"Trajectory is missing dimensions: {sorted(missing)}"
            )

        ordered = trajectory.transpose(
            "lead_hours",
            "latitude",
            "longitude",
        )

        values = np.asarray(
            ordered.values,
            dtype=np.float16,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Forecast contains NaN or infinite values."
            )

        return np.ascontiguousarray(values)