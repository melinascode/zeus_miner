"""Hourly downscaler that corrects linearly interpolated AIFS forecasts.

AIFS Single publishes 6-hourly steps but Zeus is scored hourly, so the served
forecast is a linear interpolation between bracketing steps. That interpolation
is wrong in two ways: it misses the diurnal curvature between the brackets
(largest for 2m temperature near local noon and dawn) and it inherits the
model's own bias at the bracket times. This module predicts both at once as a
gated residual on top of the interpolation, so an untrained network reproduces
plain linear interpolation exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from zeus_ml.models.lead_aware_residual_cnn import (
    LeadConditionedBlock,
    SphericalConv2d,
    _group_count,
    build_static_features,
    build_temporal_context,
)
from zeus_ml.models.lead_aware_residual_cnn_v4 import cosine_solar_zenith


VARIABLES = (
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
)
# Zeus scores t2m/u100/v100/ssrd with (0.2, 0.3, 0.3, 0.2). Dropping ssrd and
# renormalizing gives (0.25, 0.375, 0.375); we tilt slightly toward temperature
# because that is where the leaderboard gap is widest.
VARIABLE_WEIGHTS = (0.30, 0.35, 0.35)

WEATHER_CHANNELS = 6  # interpolated state (3) + bracket tendency (3)
STATIC_CHANNELS = 10
CONTEXT_FEATURES = 11
FULL_HEIGHT = 721
FULL_WIDTH = 1440
STEP_HOURS = 6
MAX_LEAD_HOURS = 360
SHORT_HORIZON_HOURS = 48.0
HORIZON_MIX_SCALE_HOURS = 24.0


@dataclass(frozen=True)
class DownscalerStatistics:
    """Per-variable normalization constants, all in physical units."""

    state_mean: tuple[float, ...]
    state_std: tuple[float, ...]
    delta_std: tuple[float, ...]
    residual_std: tuple[float, ...]

    def tensors(
        self, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def column(values: tuple[float, ...]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.float32, device=device).view(
                -1, 1, 1
            )

        return (
            column(self.state_mean),
            column(self.state_std),
            column(self.delta_std),
            column(self.residual_std),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "state_mean": list(self.state_mean),
            "state_std": list(self.state_std),
            "delta_std": list(self.delta_std),
            "residual_std": list(self.residual_std),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DownscalerStatistics":
        return cls(
            state_mean=tuple(payload["state_mean"]),
            state_std=tuple(payload["state_std"]),
            delta_std=tuple(payload["delta_std"]),
            residual_std=tuple(payload["residual_std"]),
        )


def bracket_for_lead(lead_hour: int) -> tuple[int, int, float]:
    """Return (left step index, right step index, fraction) for an hourly lead."""

    if not 0 <= lead_hour <= MAX_LEAD_HOURS:
        raise ValueError(f"lead_hour must be in [0, {MAX_LEAD_HOURS}].")
    n_steps = MAX_LEAD_HOURS // STEP_HOURS
    left = min(lead_hour // STEP_HOURS, n_steps - 1)
    fraction = (lead_hour - left * STEP_HOURS) / STEP_HOURS
    return left, left + 1, float(fraction)


def build_downscaler_context(
    lead_hour: float,
    cycle_hour: float,
    day_of_year: float,
    fraction: float,
) -> torch.Tensor:
    """Lead/hour/season context plus where we sit inside the 6-hour interval.

    The first seven entries are the shared lead-aware features, which already
    include hour-of-day and day-of-year as sin/cos pairs evaluated at the valid
    time. The last four describe the interpolation itself: `4f(1-f)` peaks
    exactly where linear interpolation is weakest and vanishes on the brackets.
    """

    base = build_temporal_context(
        torch.tensor(float(lead_hour)),
        torch.tensor(float(cycle_hour)),
        torch.tensor(float(day_of_year)),
    )
    f = float(fraction)
    two_pi = 2.0 * torch.pi
    interval = torch.tensor(
        [
            f,
            float(np.sin(two_pi * f)),
            float(np.cos(two_pi * f)),
            4.0 * f * (1.0 - f),
        ],
        dtype=torch.float32,
    )
    return torch.cat((base.to(torch.float32), interval), dim=-1)


def load_static_maps(root: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root = Path(root)
    maps = []
    for name in ("land_sea", "orography", "roughness"):
        array = np.load(root / f"{name}.npy")
        if tuple(array.shape) != (FULL_HEIGHT, FULL_WIDTH):
            raise ValueError(f"{name}.npy must have shape (721, 1440).")
        maps.append(torch.from_numpy(array).to(torch.float32))
    return maps[0], maps[1], maps[2]


def zenith_triplet(
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    cycle_time: datetime,
    lead_hour: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return solar zenith now, and its departure from the interpolated zenith.

    The second field is the cleanest available proxy for what linear
    interpolation cannot represent: solar forcing is strongly curved inside a
    6-hour window, so the gap between the true zenith and the interpolated one
    has the same shape as the temperature error we want to remove.
    """

    left, right, fraction = bracket_for_lead(lead_hour)
    utc = (
        cycle_time.replace(tzinfo=timezone.utc)
        if cycle_time.tzinfo is None
        else cycle_time.astimezone(timezone.utc)
    )
    now = cosine_solar_zenith(latitudes, longitudes, utc + timedelta(hours=lead_hour))
    z_left = cosine_solar_zenith(
        latitudes, longitudes, utc + timedelta(hours=left * STEP_HOURS)
    )
    z_right = cosine_solar_zenith(
        latitudes, longitudes, utc + timedelta(hours=right * STEP_HOURS)
    )
    interpolated = (1.0 - fraction) * z_left + fraction * z_right
    return now, now - interpolated


def build_downscaler_static_features(
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    *,
    geographic_weights: torch.Tensor,
    land_sea: torch.Tensor,
    orography: torch.Tensor,
    roughness: torch.Tensor,
    zenith: torch.Tensor,
    zenith_anomaly: torch.Tensor,
) -> torch.Tensor:
    base = build_static_features(
        latitudes,
        longitudes,
        geographic_weights=geographic_weights,
    )
    extra = torch.stack(
        (
            land_sea.to(torch.float32),
            orography.to(torch.float32),
            roughness.to(torch.float32),
            zenith.to(torch.float32),
            zenith_anomaly.to(torch.float32),
        ),
        dim=0,
    )
    if extra.shape[-2:] != base.shape[-2:]:
        raise ValueError("Static maps do not match the coordinate grid.")
    return torch.cat((base, extra), dim=0)


class DownscalerHeads(nn.Module):
    """Separate heads for temperature and the two wind components."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.temperature = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.wind = nn.Conv2d(hidden_channels, 2, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.temperature(features), self.wind(features)), dim=1)

    def zero_initialize(self) -> None:
        for head in (self.temperature, self.wind):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)


@dataclass(frozen=True)
class DownscalerOutput:
    correction: torch.Tensor
    gate: torch.Tensor
    ungated_residual: torch.Tensor
    horizon_mix: torch.Tensor


class AifsDownscalerCNN(nn.Module):
    """Gated residual corrector on top of interpolated AIFS fields."""

    def __init__(
        self,
        *,
        hidden_channels: int = 48,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 4),
        dropout: float = 0.05,
        initial_gate: float = 0.02,
        weather_channels: int = WEATHER_CHANNELS,
        static_channels: int = STATIC_CHANNELS,
        context_channels: int = CONTEXT_FEATURES,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be between zero and one.")
        self.weather_channels = weather_channels
        self.static_channels = static_channels
        self.context_channels = context_channels
        self.dilations = tuple(int(value) for value in dilations)
        self.stem = nn.Sequential(
            SphericalConv2d(weather_channels + static_channels, hidden_channels),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            LeadConditionedBlock(
                hidden_channels,
                context_channels,
                dilation=dilation,
                dropout=dropout,
            )
            for dilation in self.dilations
        )
        self.short_heads = DownscalerHeads(hidden_channels)
        self.long_heads = DownscalerHeads(hidden_channels)
        self.spatial_gate = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        self.context_gate = nn.Linear(context_channels, 3)
        self._initialize_gate(initial_gate)

    @property
    def receptive_radius(self) -> int:
        return 1 + sum(self.dilations)

    def _initialize_gate(self, initial_gate: float) -> None:
        logit = torch.logit(torch.tensor(initial_gate)).item()
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.constant_(self.spatial_gate.bias, logit)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.zeros_(self.context_gate.bias)
        # Start the correction at exactly zero so an untrained model reproduces
        # linear interpolation and training can only move away from it on
        # evidence.
        self.short_heads.zero_initialize()
        self.long_heads.zero_initialize()

    def forward(
        self,
        weather: torch.Tensor,
        context: torch.Tensor,
        static_features: torch.Tensor,
    ) -> DownscalerOutput:
        if weather.ndim != 4:
            raise ValueError("weather must have shape (batch, channel, lat, lon).")
        if context.ndim != 2 or context.shape[0] != weather.shape[0]:
            raise ValueError("context must have shape (batch, context_features).")
        if context.shape[1] != self.context_channels:
            raise ValueError(
                f"Expected {self.context_channels} context features, "
                f"received {context.shape[1]}."
            )
        if weather.shape[1] != self.weather_channels:
            raise ValueError(
                f"Expected {self.weather_channels} weather channels, "
                f"received {weather.shape[1]}."
            )
        if static_features.ndim == 3:
            static_features = static_features.unsqueeze(0).expand(
                weather.shape[0], -1, -1, -1
            )
        if static_features.shape[1] != self.static_channels:
            raise ValueError(
                f"Expected {self.static_channels} static channels, "
                f"received {static_features.shape[1]}."
            )
        if static_features.shape[2:] != weather.shape[2:]:
            raise ValueError("static feature grid does not match weather grid.")

        features = self.stem(torch.cat((weather, static_features), dim=1))
        for block in self.blocks:
            features = block(features, context)
        lead_hours = context[:, 0] * float(MAX_LEAD_HOURS)
        horizon_mix = torch.sigmoid(
            (lead_hours - SHORT_HORIZON_HOURS) / HORIZON_MIX_SCALE_HOURS
        ).view(-1, 1, 1, 1)
        residual = (1.0 - horizon_mix) * self.short_heads(features)
        residual = residual + horizon_mix * self.long_heads(features)
        gate_logits = self.spatial_gate(features)
        gate_logits = gate_logits + self.context_gate(context)[:, :, None, None]
        gate = torch.sigmoid(gate_logits)
        return DownscalerOutput(
            correction=residual * gate,
            gate=gate,
            ungated_residual=residual,
            horizon_mix=horizon_mix,
        )
