from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import xarray as xr


class WeatherDataProvider(ABC):
    """Supplies recent weather history to a forecasting model."""

    @abstractmethod
    def load_history(
        self,
        variable_name: str,
        history_hours: int,
    ) -> xr.DataArray:
        """Return recent data with a `valid_time` dimension."""
        raise NotImplementedError


class NetCDFWeatherDataProvider(WeatherDataProvider):
    """
    Loads weather history from a local NetCDF file.

    This is currently an integration provider. It will later be replaced
    by a provider that downloads current global weather data.
    """

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Weather dataset does not exist: {self.data_path}"
            )

    def load_history(
        self,
        variable_name: str,
        history_hours: int,
    ) -> xr.DataArray:
        if history_hours < 1:
            raise ValueError("history_hours must be at least 1.")

        with xr.open_dataset(self.data_path) as dataset:
            if variable_name not in dataset:
                raise KeyError(
                    f"Variable {variable_name!r} was not found in "
                    f"{self.data_path}."
                )

            data = dataset[variable_name]
            time_dimension = self._find_time_dimension(data)

            available_steps = data.sizes[time_dimension]

            if available_steps < history_hours:
                raise ValueError(
                    f"Dataset contains {available_steps} time steps, "
                    f"but {history_hours} are required."
                )

            history = data.isel(
                {time_dimension: slice(-history_hours, None)}
            ).load()

        if time_dimension != "valid_time":
            history = history.rename(
                {time_dimension: "valid_time"}
            )

        return history

    @staticmethod
    def _find_time_dimension(data: xr.DataArray) -> str:
        for candidate in ("valid_time", "time"):
            if candidate in data.dims:
                return candidate

        raise ValueError(
            "Could not find a time dimension named "
            "'valid_time' or 'time'."
        )