from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


VARIABLE_WEIGHTS = (0.2, 0.3, 0.3, 0.2)
SOLAR_CHANNEL = 3


@dataclass(frozen=True)
class ValidatorAwareLossOutput:
    loss: torch.Tensor
    corrected_combined_error: torch.Tensor
    raw_combined_error: torch.Tensor
    no_regret_penalty: torch.Tensor
    correction_penalty: torch.Tensor
    gate_penalty: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "corrected_combined_error": float(
                self.corrected_combined_error.detach()
            ),
            "raw_combined_error": float(self.raw_combined_error.detach()),
            "no_regret_penalty": float(self.no_regret_penalty.detach()),
            "correction_penalty": float(self.correction_penalty.detach()),
            "gate_penalty": float(self.gate_penalty.detach()),
        }


class ValidatorAwareResidualLoss(nn.Module):
    """Differentiable validator metric with raw-GFS regression protection."""

    def __init__(
        self,
        *,
        no_regret_weight: float = 0.5,
        correction_weight: float = 1e-4,
        gate_weight: float = 1e-4,
        epsilon: float = 1e-8,
        variable_weights: tuple[float, ...] = VARIABLE_WEIGHTS,
        solar_channel: int | None = SOLAR_CHANNEL,
        mae_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if not variable_weights:
            raise ValueError("variable_weights must not be empty.")
        if solar_channel is not None and not 0 <= solar_channel < len(
            variable_weights
        ):
            raise ValueError("solar_channel must index into variable_weights.")
        self.variable_count = len(variable_weights)
        self.solar_channel = solar_channel
        self.no_regret_weight = no_regret_weight
        self.correction_weight = correction_weight
        self.gate_weight = gate_weight
        self.epsilon = epsilon
        if not 0.0 <= mae_weight <= 1.0:
            raise ValueError("mae_weight must be in [0, 1].")
        self.mae_weight = mae_weight
        self.register_buffer(
            "variable_weights",
            torch.tensor(variable_weights, dtype=torch.float32),
        )

    def forward(
        self,
        *,
        raw_forecast: torch.Tensor,
        truth: torch.Tensor,
        correction: torch.Tensor,
        gate: torch.Tensor,
        metric_weights: torch.Tensor,
        residual_scales: torch.Tensor,
    ) -> ValidatorAwareLossOutput:
        self._validate(
            raw_forecast,
            truth,
            correction,
            gate,
            metric_weights,
            residual_scales,
            self.variable_count,
        )
        physical_correction = correction * residual_scales
        corrected = raw_forecast + physical_correction
        if self.solar_channel is not None:
            # Downwelling solar radiation cannot be negative.
            is_solar = torch.zeros(
                (1, corrected.shape[1], 1, 1),
                dtype=torch.bool,
                device=corrected.device,
            )
            is_solar[0, self.solar_channel] = True
            corrected = torch.where(is_solar, corrected.clamp_min(0.0), corrected)
        corrected_per_variable = self._combined_error(
            corrected,
            truth,
            metric_weights,
        )
        raw_per_variable = self._combined_error(
            raw_forecast,
            truth,
            metric_weights,
        )
        variable_weights = self.variable_weights.view(1, -1)
        # Errors have incompatible physical units (K, m/s, W/m²). Normalize
        # each variable by its training residual RMS before combining them.
        loss_scales = residual_scales.reshape(1, self.variable_count).clamp_min(
            self.epsilon
        )
        corrected_normalized = corrected_per_variable / loss_scales
        raw_normalized = raw_per_variable / loss_scales
        corrected_metric = (
            corrected_normalized * variable_weights
        ).sum(dim=1).mean()
        raw_metric = (raw_normalized * variable_weights).sum(dim=1).mean()

        # Per sample and variable: only penalize corrections that lose to raw.
        no_regret = torch.relu(
            corrected_normalized - raw_normalized
        )
        no_regret = (no_regret * variable_weights).sum(dim=1).mean()
        correction_penalty = correction.abs().mean()
        gate_penalty = gate.mean()
        total = (
            corrected_metric
            + self.no_regret_weight * no_regret
            + self.correction_weight * correction_penalty
            + self.gate_weight * gate_penalty
        )
        return ValidatorAwareLossOutput(
            loss=total,
            corrected_combined_error=corrected_metric,
            raw_combined_error=raw_metric,
            no_regret_penalty=no_regret,
            correction_penalty=correction_penalty,
            gate_penalty=gate_penalty,
        )

    def _combined_error(
        self,
        prediction: torch.Tensor,
        truth: torch.Tensor,
        metric_weights: torch.Tensor,
    ) -> torch.Tensor:
        error = prediction - truth
        weighted_mse = (
            error.square() * metric_weights
        ).mean(dim=(-2, -1))
        weighted_mae = (
            error.abs() * metric_weights
        ).mean(dim=(-2, -1))
        rmse = torch.sqrt(weighted_mse + self.epsilon)
        # mae_weight=0.5 reproduces the validator's (rmse + mae) / 2; larger
        # values tilt training toward the MAE half of the metric.
        return (1.0 - self.mae_weight) * rmse + self.mae_weight * weighted_mae

    @staticmethod
    def _validate(
        raw_forecast: torch.Tensor,
        truth: torch.Tensor,
        correction: torch.Tensor,
        gate: torch.Tensor,
        metric_weights: torch.Tensor,
        residual_scales: torch.Tensor,
        variable_count: int,
    ) -> None:
        expected = raw_forecast.shape
        if raw_forecast.ndim != 4 or expected[1] != variable_count:
            raise ValueError(
                f"raw_forecast must have shape (batch, {variable_count}, "
                "latitude, longitude)."
            )
        for name, value in (
            ("truth", truth),
            ("correction", correction),
            ("gate", gate),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} shape does not match raw_forecast.")
        if metric_weights.ndim != 4:
            raise ValueError(
                "metric_weights must have shape (batch, 1|4, lat, lon)."
            )
        if metric_weights.shape[0] != expected[0]:
            raise ValueError("metric_weights batch does not match forecast.")
        if metric_weights.shape[1] not in (1, variable_count):
            raise ValueError(
                f"metric_weights must have one or {variable_count} channels."
            )
        if metric_weights.shape[2:] != expected[2:]:
            raise ValueError("metric_weights grid does not match forecast.")
        if tuple(residual_scales.shape) not in (
            (variable_count,),
            (1, variable_count, 1, 1),
            (variable_count, 1, 1),
        ):
            raise ValueError(
                f"residual_scales must broadcast over {variable_count} channels."
            )
        if not all(
            torch.isfinite(value).all()
            for value in (
                raw_forecast,
                truth,
                correction,
                gate,
                metric_weights,
                residual_scales,
            )
        ):
            raise ValueError("Loss inputs must all be finite.")
