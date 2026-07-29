import xarray as xr

from .base import ForecastModel


class PersistenceForecast(ForecastModel):
    """Forecast that assumes the latest state does not change."""

    @property
    def name(self) -> str:
        return "persistence"

    def predict(
        self,
        history: xr.DataArray,
        lead_hours: int = 1,
    ) -> xr.DataArray:
        self.validate_history(history)

        if lead_hours < 1:
            raise ValueError("lead_hours must be at least 1.")

        forecast = history.isel(valid_time=-1, drop=True).copy()
        source_name = history.name or "weather_variable"
        forecast.name = f"{source_name}_forecast"
        forecast.attrs["model"] = self.name
        forecast.attrs["lead_hours"] = lead_hours
        return forecast
