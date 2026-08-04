from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from zeus.utils.coordinates import get_grid
from zeus.utils.region_mask import (
    OLD_REGION_CONFIGS,
    REGION_CONFIGS,
    build_geographic_weights,
)
from zeus.validator.constants import LATITUDE_WEIGHTS_PATH
from zeus.validator.metrics import _weighted_mae, _weighted_rmse
from zeus.validator.reward import _OLD_EUROPE_WEIGHT_CUTOFF_TS


DEFAULT_SPATIAL_SHAPE = (721, 1440)


@dataclass(frozen=True)
class ScoreResult:
    rmse: float
    mae: float
    combined_error: float
    shape_penalty: bool
    region_regime: str

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "combined_error": self.combined_error,
            "shape_penalty": self.shape_penalty,
            "region_regime": self.region_regime,
        }


class ValidatorFaithfulScorer:
    """Apply the exact Zeus validator error kernels to offline tensors."""

    def __init__(
        self,
        *,
        latitude_weights_path: str | Path = LATITUDE_WEIGHTS_PATH,
        spatial_shape: tuple[int, int] = DEFAULT_SPATIAL_SHAPE,
    ) -> None:
        self.latitude_weights_path = Path(latitude_weights_path)
        self.spatial_shape = spatial_shape

    def score(
        self,
        truth: torch.Tensor | np.ndarray,
        prediction: torch.Tensor | np.ndarray | None,
        *,
        cycle_time: datetime,
        latitude_weights: torch.Tensor | np.ndarray | None = None,
        geographic_weights: torch.Tensor | np.ndarray | None = None,
    ) -> ScoreResult:
        truth_tensor = self._to_float32(truth)
        if truth_tensor.ndim != 3:
            raise ValueError(
                f"Truth must have shape (time, latitude, longitude), "
                f"received {tuple(truth_tensor.shape)}."
            )
        if tuple(truth_tensor.shape[1:]) != self.spatial_shape:
            raise ValueError(
                f"Truth spatial shape {tuple(truth_tensor.shape[1:])} does "
                f"not match {self.spatial_shape}."
            )
        if not torch.isfinite(truth_tensor).all():
            raise ValueError("Ground truth contains NaN or Inf.")

        region_regime = self.region_regime(cycle_time)
        if prediction is None:
            return self._penalty(region_regime)
        prediction_tensor = self._to_float32(prediction)
        if (
            prediction_tensor.shape != truth_tensor.shape
            or not torch.isfinite(prediction_tensor).all()
        ):
            return self._penalty(region_regime)

        latitude_tensor = self._latitude_weights(latitude_weights)
        geographic_tensor = self._geographic_weights(
            cycle_time,
            geographic_weights,
        )
        combined_weights = (
            latitude_tensor.view(1, -1, 1)
            * geographic_tensor[None, ...]
        )
        normalized_weights = combined_weights / combined_weights.mean()
        regional_rmse = _weighted_rmse(
            prediction_tensor,
            truth_tensor,
            normalized_weights,
        )
        regional_mae = _weighted_mae(
            prediction_tensor,
            truth_tensor,
            normalized_weights,
        )
        if math.isnan(regional_rmse):
            regional_rmse = float("inf")
        if math.isnan(regional_mae):
            regional_mae = float("inf")
        combined_error = (regional_rmse + regional_mae) / 2.0
        return ScoreResult(
            rmse=regional_rmse,
            mae=regional_mae,
            combined_error=combined_error,
            shape_penalty=False,
            region_regime=region_regime,
        )

    def per_lead_diagnostics(
        self,
        truth: torch.Tensor | np.ndarray,
        prediction: torch.Tensor | np.ndarray,
        *,
        cycle_time: datetime,
    ) -> list[dict[str, float | int]]:
        """Return non-canonical per-lead diagnostics using the same spatial weights."""

        truth_tensor = self._to_float32(truth)
        prediction_tensor = self._to_float32(prediction)
        if truth_tensor.shape != prediction_tensor.shape:
            raise ValueError("Prediction and truth shapes differ.")
        if not torch.isfinite(truth_tensor).all() or not torch.isfinite(
            prediction_tensor
        ).all():
            raise ValueError("Per-lead diagnostics require finite tensors.")

        latitude = self._latitude_weights(None).view(1, -1, 1)
        geographic = self._geographic_weights(cycle_time, None)[None, ...]
        combined = latitude * geographic
        normalized = combined / combined.mean()

        rows: list[dict[str, float | int]] = []
        for lead_hour in range(truth_tensor.shape[0]):
            error = prediction_tensor[lead_hour] - truth_tensor[lead_hour]
            rmse = (
                error.square().mul(normalized[0]).mean().sqrt().item()
            )
            mae = error.abs().mul(normalized[0]).mean().item()
            rows.append(
                {
                    "lead_hour": lead_hour,
                    "regional_weighted_rmse": rmse,
                    "regional_weighted_mae": mae,
                    "combined_error": (rmse + mae) / 2.0,
                }
            )
        return rows

    @staticmethod
    def region_regime(cycle_time: datetime) -> str:
        timestamp = ValidatorFaithfulScorer._as_utc(cycle_time).timestamp()
        return (
            "europe_only"
            if timestamp < _OLD_EUROPE_WEIGHT_CUTOFF_TS
            else "europe_germany"
        )

    def _latitude_weights(
        self,
        override: torch.Tensor | np.ndarray | None,
    ) -> torch.Tensor:
        if override is None:
            values = np.load(self.latitude_weights_path)
            weights = torch.from_numpy(values).to(torch.float32)
        else:
            weights = self._to_float32(override)
        if weights.ndim != 1 or weights.shape[0] != self.spatial_shape[0]:
            raise ValueError(
                f"Latitude weights must have shape ({self.spatial_shape[0]},), "
                f"received {tuple(weights.shape)}."
            )
        if not torch.isfinite(weights).all() or weights.mean() <= 0:
            raise ValueError("Latitude weights must be finite with positive mean.")
        return weights

    def _geographic_weights(
        self,
        cycle_time: datetime,
        override: torch.Tensor | np.ndarray | None,
    ) -> torch.Tensor:
        if override is None:
            if self.spatial_shape != DEFAULT_SPATIAL_SHAPE:
                raise ValueError(
                    "Custom spatial shapes require geographic_weights."
                )
            weights = self._cached_geographic_weights(
                self.region_regime(cycle_time)
            )
        else:
            weights = self._to_float32(override)
        if tuple(weights.shape) != self.spatial_shape:
            raise ValueError(
                f"Geographic weights must have shape {self.spatial_shape}, "
                f"received {tuple(weights.shape)}."
            )
        if not torch.isfinite(weights).all() or weights.mean() <= 0:
            raise ValueError(
                "Geographic weights must be finite with positive mean."
            )
        return weights

    @staticmethod
    @lru_cache(maxsize=2)
    def _cached_geographic_weights(regime: str) -> torch.Tensor:
        grid = get_grid(-90.0, 90.0, -180.0, 179.75)
        configs = (
            OLD_REGION_CONFIGS
            if regime == "europe_only"
            else REGION_CONFIGS
        )
        return build_geographic_weights(grid, configs).contiguous()

    @staticmethod
    def _to_float32(value: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=torch.float32)
        array = np.ascontiguousarray(value)
        return torch.from_numpy(array).to(torch.float32)

    @staticmethod
    def _penalty(region_regime: str) -> ScoreResult:
        return ScoreResult(
            rmse=float("inf"),
            mae=float("inf"),
            combined_error=float("inf"),
            shape_penalty=True,
            region_regime=region_regime,
        )

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
