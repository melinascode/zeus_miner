import numpy as np
import xarray as xr

from .base import ForecastModel


class LinearTrendForecast(ForecastModel):
    """Extrapolate the recent temperature trend at every grid cell."""

    def __init__(self, window_hours: int = 6) -> None:
        if window_hours < 2:
            raise ValueError("window_hours must be at least 2.")

        self.window_hours = window_hours

    @property
    def name(self) -> str:
        return f"linear_trend_{self.window_hours}h"

    def predict(
        self,
        history: xr.DataArray,
        lead_hours: int = 1,
    ) -> xr.DataArray:
        self.validate_history(history)

        if lead_hours < 1:
            raise ValueError("lead_hours must be at least 1.")

        available_steps = history.sizes["valid_time"]

        if available_steps < self.window_hours:
            raise ValueError(
                f"{self.name} requires {self.window_hours} time steps, "
                f"but received {available_steps}."
            )

        recent = history.isel(
            valid_time=slice(-self.window_hours, None)
        )

        # Time coordinates: 0, 1, ..., window_hours - 1
        time = xr.DataArray(
            np.arange(self.window_hours, dtype=np.float32),
            dims=("valid_time",),
            coords={"valid_time": recent.valid_time},
        )

        time_mean = time.mean(dim="valid_time")
        temperature_mean = recent.mean(dim="valid_time")

        covariance = (
            (time - time_mean)
            * (recent - temperature_mean)
        ).mean(dim="valid_time")

        time_variance = (
            (time - time_mean) ** 2
        ).mean(dim="valid_time")

        slope = covariance / time_variance

        # The last observation is at time window_hours - 1.
        future_time = self.window_hours - 1 + lead_hours

        forecast = temperature_mean + slope * (
            future_time - time_mean
        )

        forecast.name = "t2m_forecast"
        forecast.attrs["model"] = self.name
        forecast.attrs["lead_hours"] = lead_hours
        forecast.attrs["window_hours"] = self.window_hours

        return forecast