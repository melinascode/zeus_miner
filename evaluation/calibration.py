from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from evaluation.selection import canonical_json_bytes, content_sha256
from zeus.validator.constants import LATITUDE_WEIGHTS_PATH


SUPPORTED_VARIABLES = (
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
)
CYCLE_HOURS = (0, 6, 12, 18)
LONG_HORIZON_HOURS = 360
LONG_REQUESTED_HOURS = LONG_HORIZON_HOURS + 1
FORMULA_TYPE = "additive_lead_hour_bias_synoptic_stratified"


@dataclass
class BiasAccumulator:
    """Accumulate pooled latitude-weighted bias (ERA5 − raw_GFS) on calib only.

    For each lead and synoptic hour::

        numerator   = Σ mask · latitude_weight · (truth − forecast)
        denominator = Σ mask · latitude_weight
        bias        = numerator / denominator

    Sums run over all calibration cycles, latitudes, and longitudes with
    ``mask`` true where both forecast and truth are finite.
    """

    latitude_weights_path: Path = LATITUDE_WEIGHTS_PATH
    n_leads: int = LONG_REQUESTED_HOURS
    _latitude_weights: np.ndarray | None = field(default=None, init=False, repr=False)
    _numerators: dict[str, dict[int, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )
    _denominators: dict[str, dict[int, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )
    _counts: dict[str, dict[int, int]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.latitude_weights_path = Path(self.latitude_weights_path)
        for variable in SUPPORTED_VARIABLES:
            self._numerators[variable] = {
                hour: np.zeros(self.n_leads, dtype=np.float64)
                for hour in CYCLE_HOURS
            }
            self._denominators[variable] = {
                hour: np.zeros(self.n_leads, dtype=np.float64)
                for hour in CYCLE_HOURS
            }
            self._counts[variable] = {hour: 0 for hour in CYCLE_HOURS}

    @property
    def latitude_weights(self) -> np.ndarray:
        if self._latitude_weights is None:
            weights = np.load(self.latitude_weights_path).astype(np.float64)
            if weights.ndim != 1 or weights.shape[0] < 1:
                raise ValueError("Latitude weights must be a 1-D array.")
            if not np.isfinite(weights).all() or weights.mean() <= 0:
                raise ValueError(
                    "Latitude weights must be finite with positive mean."
                )
            self._latitude_weights = weights
        return self._latitude_weights

    def update(
        self,
        *,
        variable: str,
        cycle_time: datetime,
        forecast: torch.Tensor | np.ndarray,
        truth: torch.Tensor | np.ndarray,
    ) -> None:
        if variable not in SUPPORTED_VARIABLES:
            raise ValueError(f"Unsupported calibration variable: {variable}")
        cycle_utc = _as_utc(cycle_time)
        cycle_hour = cycle_utc.hour
        if cycle_hour not in CYCLE_HOURS:
            raise ValueError(
                f"Calibration cycle hour must be one of {CYCLE_HOURS}."
            )
        forecast_array = _as_float64(forecast)
        truth_array = _as_float64(truth)
        if forecast_array.shape != truth_array.shape:
            raise ValueError(
                f"Forecast shape {forecast_array.shape} does not match "
                f"truth shape {truth_array.shape}."
            )
        if forecast_array.ndim != 3:
            raise ValueError(
                "Calibration tensors must have shape "
                "(time, latitude, longitude)."
            )
        if forecast_array.shape[0] != self.n_leads:
            raise ValueError(
                f"Expected {self.n_leads} leads, received "
                f"{forecast_array.shape[0]}."
            )
        if forecast_array.shape[1] != self.latitude_weights.shape[0]:
            raise ValueError(
                "Forecast latitude dimension does not match latitude weights."
            )
        mask = np.isfinite(forecast_array) & np.isfinite(truth_array)
        if not mask.any():
            raise ValueError("Calibration tensors have no valid (finite) cells.")

        residual = np.where(mask, truth_array - forecast_array, 0.0)
        weights = self.latitude_weights.reshape(1, -1, 1)
        sample_weight = mask.astype(np.float64) * weights
        self._numerators[variable][cycle_hour] += (residual * sample_weight).sum(
            axis=(1, 2)
        )
        self._denominators[variable][cycle_hour] += sample_weight.sum(axis=(1, 2))
        self._counts[variable][cycle_hour] += 1

    def per_cycle_denominators(self) -> dict[str, dict[int, np.ndarray]]:
        """Return per-lead spatial weight sums for each variable and synoptic hour."""
        return {
            variable: {
                hour: self._denominators[variable][hour].copy()
                for hour in CYCLE_HOURS
            }
            for variable in SUPPORTED_VARIABLES
        }

    def seed_from_frozen(
        self,
        frozen: "FrozenCalibration",
        *,
        per_cycle_denominators: Mapping[str, Mapping[int, np.ndarray]],
    ) -> dict[str, list[str]]:
        """Restore running sums from frozen means and one-cycle denominators."""
        payload = frozen.payload
        fitted_issue_cycles = payload.get("fitted_issue_cycles") or {}
        fitted: dict[str, list[str]] = {}
        for variable in SUPPORTED_VARIABLES:
            cycles = list(fitted_issue_cycles.get(variable, []))
            if not cycles:
                raise ValueError(
                    f"Frozen coefficients missing fitted_issue_cycles for {variable}."
                )
            fitted[variable] = cycles
            for hour in CYCLE_HOURS:
                count = int(payload["sample_counts"][variable][str(hour)])
                if count < 1:
                    raise ValueError(
                        f"No frozen sample count for {variable} at {hour:02d}Z."
                    )
                mean = np.asarray(
                    payload["biases"][variable][str(hour)],
                    dtype=np.float64,
                )
                den_per_cycle = np.asarray(
                    per_cycle_denominators[variable][hour],
                    dtype=np.float64,
                )
                if den_per_cycle.shape != mean.shape:
                    raise ValueError(
                        f"Denominator shape {den_per_cycle.shape} does not match "
                        f"bias shape {mean.shape} for {variable} at {hour:02d}Z."
                    )
                total_den = den_per_cycle * count
                self._numerators[variable][hour] = mean * total_den
                self._denominators[variable][hour] = total_den
                self._counts[variable][hour] = count
        return fitted

    def freeze(
        self,
        *,
        selection_sha256: str,
        plan_id: str = "benchmark_v1",
        fitted_issue_cycles: Mapping[str, list[str]] | None = None,
    ) -> "FrozenCalibration":
        biases: dict[str, dict[str, list[float]]] = {}
        sample_counts: dict[str, dict[str, int]] = {}
        for variable in SUPPORTED_VARIABLES:
            biases[variable] = {}
            sample_counts[variable] = {}
            for hour in CYCLE_HOURS:
                count = self._counts[variable][hour]
                if count < 1:
                    raise ValueError(
                        f"No calibration samples for {variable} at "
                        f"{hour:02d}Z."
                    )
                denominator = self._denominators[variable][hour]
                if np.any(denominator <= 0.0):
                    raise ValueError(
                        f"Zero weighted mass for {variable} at {hour:02d}Z."
                    )
                mean = self._numerators[variable][hour] / denominator
                biases[variable][str(hour)] = [
                    float(value) for value in mean.tolist()
                ]
                sample_counts[variable][str(hour)] = count

        payload = {
            "schema_version": 1,
            "plan_id": plan_id,
            "selection_sha256": selection_sha256,
            "formula": {
                "type": FORMULA_TYPE,
                "weights": "latitude_only",
                "region_weights": False,
                "solar_clip_min": 0.0,
                "expression": (
                    "b = Σ mask·L·(ERA5−raw_GFS) / Σ mask·L; "
                    "F_cal = F_raw + b[V,h,c]; solar: max(0, F_cal)"
                ),
                "pooling": "global_over_calib_cycles_lats_lons",
            },
            "variables": list(SUPPORTED_VARIABLES),
            "cycle_hours": list(CYCLE_HOURS),
            "n_leads": self.n_leads,
            "horizon_hours": LONG_HORIZON_HOURS,
            "biases": biases,
            "sample_counts": sample_counts,
            "fitted_issue_cycles": fitted_issue_cycles or {},
            "latitude_weights_path": str(self.latitude_weights_path),
            "frozen_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        # Round-trip through JSON before hashing so load-time verification matches.
        round_tripped = json.loads(canonical_json_bytes(payload).decode("utf-8"))
        round_tripped["coefficients_sha256"] = content_sha256(
            round_tripped,
            digest_key="coefficients_sha256",
        )
        return FrozenCalibration(round_tripped)


@dataclass(frozen=True)
class FrozenCalibration:
    payload: dict[str, Any]

    @property
    def selection_sha256(self) -> str:
        return str(self.payload["selection_sha256"])

    @property
    def coefficients_sha256(self) -> str:
        return str(self.payload["coefficients_sha256"])

    def bias(
        self,
        variable: str,
        cycle_hour: int,
        n_leads: int,
    ) -> np.ndarray:
        if variable not in self.payload["biases"]:
            raise KeyError(f"No frozen bias for variable {variable}.")
        values = self.payload["biases"][variable][str(cycle_hour)]
        if n_leads > len(values):
            raise ValueError(
                f"Requested {n_leads} leads but coefficients only have "
                f"{len(values)}."
            )
        return np.asarray(values[:n_leads], dtype=np.float32)

    def write(
        self,
        path: str | Path,
        *,
        allow_overwrite: bool = False,
    ) -> Path:
        destination = Path(path)
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing == self.payload:
                return destination
            if not allow_overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite different coefficients: {destination}"
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = canonical_json_bytes(self.payload)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with temporary.open("wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_selection_sha256: str | None = None,
    ) -> "FrozenCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        digest = content_sha256(
            payload,
            digest_key="coefficients_sha256",
        )
        if payload.get("coefficients_sha256") != digest:
            raise ValueError(
                f"coefficients_sha256 mismatch for {path}: "
                f"recorded={payload.get('coefficients_sha256')} "
                f"computed={digest}"
            )
        if payload.get("formula", {}).get("type") != FORMULA_TYPE:
            raise ValueError("Unsupported calibration formula type.")
        if (
            expected_selection_sha256 is not None
            and payload.get("selection_sha256") != expected_selection_sha256
        ):
            raise ValueError(
                "Frozen coefficients were fit against a different selection."
            )
        return cls(payload)


def assert_no_test_cycle_in_fit(
    fitted_issue_cycles: Mapping[str, list[str]] | list[str],
    test_cycles: list[str],
) -> None:
    if isinstance(fitted_issue_cycles, Mapping):
        fitted = {
            cycle
            for cycles in fitted_issue_cycles.values()
            for cycle in cycles
        }
    else:
        fitted = set(fitted_issue_cycles)
    overlap = fitted.intersection(test_cycles)
    if overlap:
        raise ValueError(
            "Calibration fit includes test cycles (leakage): "
            f"{sorted(overlap)}"
        )


def _as_float64(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().to(device="cpu").numpy()
    else:
        array = np.asarray(value)
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
