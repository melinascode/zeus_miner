from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIABLES = (
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
)
CONTEXT_FEATURES = 7
STATIC_CHANNELS = 5


def build_temporal_context(
    lead_hours: torch.Tensor,
    cycle_hours: torch.Tensor,
    day_of_year: torch.Tensor,
) -> torch.Tensor:
    """Build bounded lead, valid-hour, and seasonal context features."""

    lead = lead_hours.to(torch.float32)
    cycle = cycle_hours.to(torch.float32)
    day = day_of_year.to(torch.float32)
    valid_hour = torch.remainder(cycle + lead, 24.0)
    valid_day = torch.remainder(day - 1.0 + lead / 24.0, 365.2425)
    two_pi = 2.0 * torch.pi
    return torch.stack(
        (
            lead / 360.0,
            torch.sin(two_pi * lead / 360.0),
            torch.cos(two_pi * lead / 360.0),
            torch.sin(two_pi * valid_hour / 24.0),
            torch.cos(two_pi * valid_hour / 24.0),
            torch.sin(two_pi * valid_day / 365.2425),
            torch.cos(two_pi * valid_day / 365.2425),
        ),
        dim=-1,
    )


def context_from_cycle(
    cycle_time: datetime,
    lead_hours: torch.Tensor,
) -> torch.Tensor:
    cycle = (
        cycle_time.replace(tzinfo=timezone.utc)
        if cycle_time.tzinfo is None
        else cycle_time.astimezone(timezone.utc)
    )
    shape = lead_hours.shape
    cycle_hours = torch.full(
        shape,
        float(cycle.hour),
        dtype=torch.float32,
        device=lead_hours.device,
    )
    day_of_year = torch.full(
        shape,
        float(cycle.timetuple().tm_yday),
        dtype=torch.float32,
        device=lead_hours.device,
    )
    return build_temporal_context(lead_hours, cycle_hours, day_of_year)


def build_static_features(
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    *,
    geographic_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return sin/cos coordinates and normalized challenge weight channels."""

    latitudes = latitudes.to(torch.float32)
    longitudes = longitudes.to(torch.float32)
    if latitudes.ndim != 1 or longitudes.ndim != 1:
        raise ValueError("latitudes and longitudes must be one-dimensional.")
    lat_rad = torch.deg2rad(latitudes).view(-1, 1)
    lon_rad = torch.deg2rad(longitudes).view(1, -1)
    height = latitudes.numel()
    width = longitudes.numel()
    sin_lat = torch.sin(lat_rad).expand(height, width)
    cos_lat = torch.cos(lat_rad).expand(height, width)
    sin_lon = torch.sin(lon_rad).expand(height, width)
    cos_lon = torch.cos(lon_rad).expand(height, width)
    if geographic_weights is None:
        region = torch.ones((height, width), dtype=torch.float32)
    else:
        region = geographic_weights.to(torch.float32)
        if tuple(region.shape) != (height, width):
            raise ValueError(
                "geographic_weights shape does not match coordinate grid."
            )
        region = region / region.mean()
    return torch.stack((sin_lat, cos_lat, sin_lon, cos_lon, region), dim=0)


class SphericalConv2d(nn.Module):
    """Convolution with circular longitude and reflected latitude padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("SphericalConv2d requires an odd kernel size.")
        self.padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.padding
        if pad:
            x = F.pad(x, (pad, pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, pad, pad), mode="reflect")
        return self.conv(x)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class LeadConditionedBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        context_channels: int,
        *,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = SphericalConv2d(
            channels,
            channels,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.film = nn.Linear(context_channels, channels * 2)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.norm(x)
        scale, shift = self.film(context).chunk(2, dim=-1)
        residual = residual * (1.0 + scale[:, :, None, None])
        residual = residual + shift[:, :, None, None]
        residual = F.silu(residual)
        residual = self.depthwise(residual)
        residual = F.silu(self.pointwise(residual))
        return x + self.dropout(residual)


@dataclass(frozen=True)
class GatedResidualOutput:
    correction: torch.Tensor
    gate: torch.Tensor
    ungated_residual: torch.Tensor


class LeadAwareGatedResidualCNN(nn.Module):
    """Lead-aware, no-regret residual corrector for the four Zeus variables."""

    def __init__(
        self,
        *,
        weather_channels: int = 4,
        static_channels: int = STATIC_CHANNELS,
        context_channels: int = CONTEXT_FEATURES,
        hidden_channels: int = 32,
        dilations: Sequence[int] = (1, 2, 4, 8, 4, 2),
        dropout: float = 0.05,
        initial_gate: float = 0.02,
    ) -> None:
        super().__init__()
        if weather_channels != 4:
            raise ValueError("The current variable heads require four channels.")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be between zero and one.")
        self.weather_channels = weather_channels
        self.static_channels = static_channels
        self.context_channels = context_channels
        self.hidden_channels = hidden_channels
        self.dilations = tuple(int(value) for value in dilations)
        self.stem = nn.Sequential(
            SphericalConv2d(
                weather_channels + static_channels,
                hidden_channels,
            ),
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
        self.temperature_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.wind_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.solar_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.spatial_gate = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        self.context_gate = nn.Linear(context_channels, 4)
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

    def forward(
        self,
        weather: torch.Tensor,
        context: torch.Tensor,
        static_features: torch.Tensor,
    ) -> GatedResidualOutput:
        if weather.ndim != 4:
            raise ValueError("weather must have shape (batch, channel, lat, lon).")
        if context.ndim != 2 or context.shape[0] != weather.shape[0]:
            raise ValueError("context must have shape (batch, context_features).")
        if context.shape[1] != self.context_channels:
            raise ValueError(
                f"Expected {self.context_channels} context features, "
                f"received {context.shape[1]}."
            )
        if static_features.ndim == 3:
            static_features = static_features.unsqueeze(0).expand(
                weather.shape[0], -1, -1, -1
            )
        if static_features.ndim != 4:
            raise ValueError(
                "static_features must have shape (static, lat, lon) or "
                "(batch, static, lat, lon)."
            )
        if static_features.shape[0] != weather.shape[0]:
            raise ValueError("static feature batch does not match weather batch.")
        if static_features.shape[2:] != weather.shape[2:]:
            raise ValueError("static feature grid does not match weather grid.")
        if weather.shape[1] != self.weather_channels:
            raise ValueError(
                f"Expected {self.weather_channels} weather channels, "
                f"received {weather.shape[1]}."
            )
        if static_features.shape[1] != self.static_channels:
            raise ValueError(
                f"Expected {self.static_channels} static channels, "
                f"received {static_features.shape[1]}."
            )

        features = self.stem(torch.cat((weather, static_features), dim=1))
        for block in self.blocks:
            features = block(features, context)
        residual = torch.cat(
            (
                self.temperature_head(features),
                self.wind_head(features),
                self.solar_head(features),
            ),
            dim=1,
        )
        gate_logits = self.spatial_gate(features)
        gate_logits = gate_logits + self.context_gate(context)[:, :, None, None]
        gate = torch.sigmoid(gate_logits)
        return GatedResidualOutput(
            correction=residual * gate,
            gate=gate,
            ungated_residual=residual,
        )
