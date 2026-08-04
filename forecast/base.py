from abc import ABC, abstractmethod

import xarray as xr


class ForecastModel(ABC):
    """Common interface implemented by every forecasting engine."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        history: xr.DataArray,
        lead_hours: int = 1,
    ) -> xr.DataArray:
        """
        Produce a forecast from historical weather data.

        Parameters
        ----------
        history:
            Temperature history with dimensions:
            (valid_time, latitude, longitude)

        lead_hours:
            Number of hours into the future to predict.

        Returns
        -------
        xr.DataArray
            Forecast with dimensions:
            (latitude, longitude)
        """
        raise NotImplementedError

    def validate_history(self, history: xr.DataArray) -> None:
        required_dimensions = {"valid_time", "latitude", "longitude"}

        missing = required_dimensions.difference(history.dims)

        if missing:
            raise ValueError(
                f"History is missing required dimensions: {sorted(missing)}"
            )

        if history.sizes["valid_time"] < 1:
            raise ValueError("History must contain at least one time step.")