from __future__ import annotations

import numpy as np
import xarray as xr

from forecast.base import ForecastModel
from forecast.data_provider import WeatherDataProvider
from forecast.persistence import PersistenceForecast
from forecast.trajectory import ForecastTrajectoryService
from forecast.variables import canonicalize_variable_name


class ForecastService:
    """Produce one Zeus-compatible variable trajectory at a time."""

    SHORT_FORECAST_STEPS = 49
    LONG_FORECAST_STEPS = 361
    EXPECTED_LATITUDE_SIZE = 721
    EXPECTED_LONGITUDE_SIZE = 1440

    def __init__(
        self,
        data_provider: WeatherDataProvider,
        variable_name: str = "2m_temperature",
        history_hours: int = 24,
        model: ForecastModel | None = None,
    ) -> None:
        if history_hours < 1:
            raise ValueError("history_hours must be at least 1.")

        self.data_provider = data_provider
        self.variable_name = canonicalize_variable_name(variable_name)
        self.history_hours = history_hours
        self.model = model or PersistenceForecast()
        self.trajectory_service = ForecastTrajectoryService(model=self.model)

    @property
    def model_name(self) -> str:
        return self.model.name

    def load_history(self) -> xr.DataArray:
        return self.data_provider.load_history(
            variable_name=self.variable_name,
            history_hours=self.history_hours,
        )

    def generate_from_history(
        self,
        history: xr.DataArray,
        forecast_steps: int,
    ) -> np.ndarray:
        if forecast_steps < 1:
            raise ValueError("forecast_steps must be at least 1.")

        # Persistence is the protocol baseline. Building hundreds of xarray
        # objects would temporarily use several gigabytes, so repeat the latest
        # field directly into one float16 C-contiguous tensor.
        if isinstance(self.model, PersistenceForecast):
            latest = history.isel(valid_time=-1, drop=True).transpose(
                "latitude", "longitude"
            )
            base = np.ascontiguousarray(latest.values, dtype=np.float16)
            forecast = np.empty(
                (
                    forecast_steps,
                    self.EXPECTED_LATITUDE_SIZE,
                    self.EXPECTED_LONGITUDE_SIZE,
                ),
                dtype=np.float16,
            )
            forecast[:] = base
        else:
            trajectory = self.trajectory_service.predict(
                history=history,
                forecast_hours=forecast_steps,
                include_initial_state=True,
            )
            forecast = self.trajectory_service.to_zeus_array(trajectory)

        self._validate_forecast(forecast, forecast_steps)
        return forecast

    def _generate(
        self,
        forecast_steps: int,
        history: xr.DataArray | None = None,
    ) -> np.ndarray:
        loaded_history = history if history is not None else self.load_history()
        return self.generate_from_history(loaded_history, forecast_steps)

    def generate_short_forecast(
        self,
        history: xr.DataArray | None = None,
    ) -> np.ndarray:
        return self._generate(self.SHORT_FORECAST_STEPS, history=history)

    def generate_long_forecast(
        self,
        history: xr.DataArray | None = None,
    ) -> np.ndarray:
        return self._generate(self.LONG_FORECAST_STEPS, history=history)

    @classmethod
    def _validate_forecast(
        cls,
        forecast: np.ndarray,
        expected_steps: int,
    ) -> None:
        expected_shape = (
            expected_steps,
            cls.EXPECTED_LATITUDE_SIZE,
            cls.EXPECTED_LONGITUDE_SIZE,
        )
        if forecast.shape != expected_shape:
            raise ValueError(
                f"Expected forecast shape {expected_shape}, received "
                f"{forecast.shape}."
            )
        if forecast.dtype != np.float16:
            raise TypeError(
                f"Expected float16 forecast, received {forecast.dtype}."
            )
        if not forecast.flags.c_contiguous:
            raise ValueError("Forecast array is not C-contiguous.")
        if not np.isfinite(forecast).all():
            raise ValueError("Forecast contains NaN or infinite values.")
