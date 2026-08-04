from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class ForecastMetrics:
    rmse: float
    weighted_rmse: float
    mae: float
    bias: float
    maximum_absolute_error: float


class ForecastEvaluator:
    """Compare a forecast against ERA5 ground truth."""

    @staticmethod
    def _align(
        forecast: xr.DataArray,
        truth: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        forecast, truth = xr.align(
            forecast,
            truth,
            join="exact",
        )

        return forecast, truth

    @staticmethod
    def error_field(
        forecast: xr.DataArray,
        truth: xr.DataArray,
    ) -> xr.DataArray:
        forecast, truth = ForecastEvaluator._align(forecast, truth)

        error = forecast - truth
        error.name = "forecast_error"

        return error

    @staticmethod
    def latitude_weighted_rmse(
        forecast: xr.DataArray,
        truth: xr.DataArray,
    ) -> float:
        forecast, truth = ForecastEvaluator._align(forecast, truth)

        squared_error = (forecast - truth) ** 2

        latitude_weights = np.cos(
            np.deg2rad(forecast.latitude)
        )

        weighted_mse = squared_error.weighted(
            latitude_weights
        ).mean(
            dim=("latitude", "longitude")
        )

        return float(np.sqrt(weighted_mse.values))

    def evaluate(
        self,
        forecast: xr.DataArray,
        truth: xr.DataArray,
    ) -> ForecastMetrics:
        error = self.error_field(forecast, truth)

        rmse = float(np.sqrt((error**2).mean().values))
        weighted_rmse = self.latitude_weighted_rmse(
            forecast,
            truth,
        )
        mae = float(np.abs(error).mean().values)
        bias = float(error.mean().values)
        maximum_absolute_error = float(
            np.abs(error).max().values
        )

        return ForecastMetrics(
            rmse=rmse,
            weighted_rmse=weighted_rmse,
            mae=mae,
            bias=bias,
            maximum_absolute_error=maximum_absolute_error,
        )