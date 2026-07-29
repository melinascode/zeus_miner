import xarray as xr

from .base import ForecastModel


class MovingAverageForecast(ForecastModel):
    """
    Forecast using the mean of the most recent observations.

    This smooths short-term noise but may lag rapid temperature changes.
    """

    def __init__(self, window_hours: int = 6) -> None:
        if window_hours < 1:
            raise ValueError("window_hours must be at least 1.")

        self.window_hours = window_hours

    @property
    def name(self) -> str:
        return f"moving_average_{self.window_hours}h"

    def predict(
        self,
        history: xr.DataArray,
        lead_hours: int = 1,
    ) -> xr.DataArray:
        self.validate_history(history)

        if lead_hours < 1:
            raise ValueError("lead_hours must be at least 1.")

        available_hours = history.sizes["valid_time"]

        if available_hours < self.window_hours:
            raise ValueError(
                f"{self.name} requires at least "
                f"{self.window_hours} history steps, "
                f"but received {available_hours}."
            )

        recent_history = history.isel(
            valid_time=slice(-self.window_hours, None)
        )

        forecast = recent_history.mean(
            dim="valid_time",
            keep_attrs=True,
        )

        forecast.name = "t2m_forecast"
        forecast.attrs["model"] = self.name
        forecast.attrs["lead_hours"] = lead_hours
        forecast.attrs["window_hours"] = self.window_hours

        return forecast