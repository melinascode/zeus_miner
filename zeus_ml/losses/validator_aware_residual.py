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
    ) -> None:
        super().__init__()
        self.no_regret_weight = no_regret_weight
        self.correction_weight = correction_weight
        self.gate_weight = gate_weight
        self.epsilon = epsilon
        self.register_buffer(
            "variable_weights",
            torch.tensor(VARIABLE_WEIGHTS, dtype=torch.float32),
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
        )
        physical_correction = correction * residual_scales
        corrected = raw_forecast + physical_correction
        corrected_solar = corrected[:, SOLAR_CHANNEL].clamp_min(0.0)
        corrected = torch.cat(
            (
                corrected[:, :SOLAR_CHANNEL],
                corrected_solar.unsqueeze(1),
            ),
            dim=1,
        )
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
        loss_scales = residual_scales.reshape(1, 4).clamp_min(self.epsilon)
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
        return (rmse + weighted_mae) / 2.0

    @staticmethod
    def _validate(
        raw_forecast: torch.Tensor,
        truth: torch.Tensor,
        correction: torch.Tensor,
        gate: torch.Tensor,
        metric_weights: torch.Tensor,
        residual_scales: torch.Tensor,
    ) -> None:
        expected = raw_forecast.shape
        if raw_forecast.ndim != 4 or expected[1] != 4:
            raise ValueError(
                "raw_forecast must have shape (batch, 4, latitude, longitude)."
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
        if metric_weights.shape[1] not in (1, 4):
            raise ValueError("metric_weights must have one or four channels.")
        if metric_weights.shape[2:] != expected[2:]:
            raise ValueError("metric_weights grid does not match forecast.")
        if tuple(residual_scales.shape) not in (
            (4,),
            (1, 4, 1, 1),
            (4, 1, 1),
        ):
            raise ValueError("residual_scales must broadcast over four channels.")
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
