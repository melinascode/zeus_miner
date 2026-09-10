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
    gradient_penalty: torch.Tensor | None = None
    speed_penalty: torch.Tensor | None = None

    def detached_metrics(self) -> dict[str, float]:
        metrics = {
            "loss": float(self.loss.detach()),
            "corrected_combined_error": float(
                self.corrected_combined_error.detach()
            ),
            "raw_combined_error": float(self.raw_combined_error.detach()),
            "no_regret_penalty": float(self.no_regret_penalty.detach()),
            "correction_penalty": float(self.correction_penalty.detach()),
            "gate_penalty": float(self.gate_penalty.detach()),
        }
        if self.gradient_penalty is not None:
            metrics["gradient_penalty"] = float(self.gradient_penalty.detach())
        if self.speed_penalty is not None:
            metrics["speed_penalty"] = float(self.speed_penalty.detach())
        return metrics


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
        gradient_weight: float = 0.0,
        speed_weight: float = 0.0,
        wind_channels: tuple[int, int] | None = None,
        wind_no_regret_weight: float = 0.0,
        gradient_wind_only: bool = True,
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
        if gradient_weight < 0.0:
            raise ValueError("gradient_weight must be non-negative.")
        self.gradient_weight = gradient_weight
        if speed_weight < 0.0:
            raise ValueError("speed_weight must be non-negative.")
        if speed_weight > 0.0 and wind_channels is None:
            raise ValueError("wind_channels is required when speed_weight > 0.")
        if wind_no_regret_weight < 0.0:
            raise ValueError("wind_no_regret_weight must be non-negative.")
        if wind_channels is not None:
            if len(wind_channels) != 2 or not all(
                0 <= c < len(variable_weights) for c in wind_channels
            ):
                raise ValueError("wind_channels must be two valid channel indices.")
        self.speed_weight = speed_weight
        self.wind_channels = tuple(wind_channels) if wind_channels else None
        self.wind_no_regret_weight = wind_no_regret_weight
        self.gradient_wind_only = bool(gradient_wind_only)
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
        no_regret_per = torch.relu(corrected_normalized - raw_normalized)
        no_regret = (no_regret_per * variable_weights).sum(dim=1).mean()
        if (
            self.wind_no_regret_weight > 0.0
            and self.wind_channels is not None
        ):
            u_idx, v_idx = self.wind_channels
            wind_nr = 0.5 * (no_regret_per[:, u_idx] + no_regret_per[:, v_idx])
            no_regret = no_regret + self.wind_no_regret_weight * wind_nr.mean()
        correction_penalty = correction.abs().mean()
        gate_penalty = gate.mean()

        # Gradient sharpness: served fields carry only ~80% of ERA5's wind
        # gradient energy, an over-smoothing the point metric barely sees.
        # Match the truth's spatial gradients directly (metric-weighted MAE
        # on lat/lon finite differences, normalized like the point terms).
        gradient_penalty = None
        if self.gradient_weight > 0.0:
            gradient_penalty = self._gradient_error(
                corrected, truth, metric_weights
            )
            gradient_normalized = gradient_penalty / loss_scales
            grad_weights = variable_weights
            if self.gradient_wind_only and self.wind_channels is not None:
                mask = torch.zeros_like(variable_weights)
                for idx in self.wind_channels:
                    mask[:, idx] = variable_weights[:, idx]
                grad_weights = mask
            gradient_penalty = (
                (gradient_normalized * grad_weights).sum(dim=1).mean()
            )

        # Wind speed: the ensemble mean damps |V| (averaging members with
        # different directions shrinks the vector), so u/v can be unbiased
        # while speed is low. Match the truth's speed field directly.
        speed_penalty = None
        if self.speed_weight > 0.0 and self.wind_channels is not None:
            u_idx, v_idx = self.wind_channels
            speed_pred = torch.sqrt(
                corrected[:, u_idx].square()
                + corrected[:, v_idx].square()
                + self.epsilon
            )
            speed_true = torch.sqrt(
                truth[:, u_idx].square() + truth[:, v_idx].square() + self.epsilon
            )
            w = metric_weights[:, min(u_idx, metric_weights.shape[1] - 1)]
            speed_err = ((speed_pred - speed_true).abs() * w).mean(dim=(-2, -1))
            wind_scale = 0.5 * (loss_scales[0, u_idx] + loss_scales[0, v_idx])
            speed_penalty = (speed_err / wind_scale).mean()

        total = (
            corrected_metric
            + self.no_regret_weight * no_regret
            + self.correction_weight * correction_penalty
            + self.gate_weight * gate_penalty
        )
        if gradient_penalty is not None:
            total = total + self.gradient_weight * gradient_penalty
        if speed_penalty is not None:
            total = total + self.speed_weight * speed_penalty
        return ValidatorAwareLossOutput(
            loss=total,
            corrected_combined_error=corrected_metric,
            raw_combined_error=raw_metric,
            no_regret_penalty=no_regret,
            correction_penalty=correction_penalty,
            gate_penalty=gate_penalty,
            gradient_penalty=gradient_penalty,
            speed_penalty=speed_penalty,
        )

    @staticmethod
    def _gradient_error(
        prediction: torch.Tensor,
        truth: torch.Tensor,
        metric_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Metric-weighted MAE of spatial gradients, per sample and variable."""

        d_pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
        d_true_x = truth[..., :, 1:] - truth[..., :, :-1]
        d_pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
        d_true_y = truth[..., 1:, :] - truth[..., :-1, :]
        err_x = (
            (d_pred_x - d_true_x).abs() * metric_weights[..., :, 1:]
        ).mean(dim=(-2, -1))
        err_y = (
            (d_pred_y - d_true_y).abs() * metric_weights[..., 1:, :]
        ).mean(dim=(-2, -1))
        return err_x + err_y

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
