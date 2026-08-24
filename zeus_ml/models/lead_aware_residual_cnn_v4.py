from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from zeus_ml.models.lead_aware_residual_cnn import (
    CONTEXT_FEATURES,
    FULL_LATITUDE_SIZE,
    GatedResidualOutput,
    LeadAwareGatedResidualCNN,
    build_static_features,
    context_from_cycle,
)


COARSE_HEIGHT = 91
COARSE_WIDTH = 180
V4_STATIC_CHANNELS = 8
NATIVE_SHAPE = (FULL_LATITUDE_SIZE, 1440)


def cosine_solar_zenith(
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    valid_time: datetime,
) -> torch.Tensor:
    """Return cosine of solar zenith on a lat/lon grid. Night is negative."""

    utc = (
        valid_time.replace(tzinfo=timezone.utc)
        if valid_time.tzinfo is None
        else valid_time.astimezone(timezone.utc)
    )
    day = float(utc.timetuple().tm_yday)
    hour = utc.hour + utc.minute / 60.0 + utc.second / 3600.0
    declination = 23.44 * np.sin(2.0 * np.pi * (day - 81.0) / 365.2425)
    lat = torch.deg2rad(latitudes.to(torch.float32)).view(-1, 1)
    lon = longitudes.to(torch.float32).view(1, -1)
    hour_angle = torch.deg2rad(15.0 * (hour - 12.0) + lon)
    decl = torch.tensor(np.deg2rad(declination), dtype=torch.float32)
    return (
        torch.sin(lat) * torch.sin(decl)
        + torch.cos(lat) * torch.cos(decl) * torch.cos(hour_angle)
    ).clamp(-1.0, 1.0)


def downsample_2deg(field: torch.Tensor) -> torch.Tensor:
    """Area-average a native 0.25° field to about 2°."""

    if field.ndim == 3:
        field = field.unsqueeze(0)
        squeeze = True
    elif field.ndim == 4:
        squeeze = False
    else:
        raise ValueError("Expected (channel, lat, lon) or (batch, channel, lat, lon).")
    coarse = F.interpolate(
        field.to(torch.float32),
        size=(COARSE_HEIGHT, COARSE_WIDTH),
        mode="area",
    )
    return coarse.squeeze(0) if squeeze else coarse


def upsample_native(field: torch.Tensor) -> torch.Tensor:
    if field.ndim == 3:
        field = field.unsqueeze(0)
        squeeze = True
    elif field.ndim == 4:
        squeeze = False
    else:
        raise ValueError("Expected (channel, lat, lon) or (batch, channel, lat, lon).")
    native = F.interpolate(
        field.to(torch.float32),
        size=NATIVE_SHAPE,
        mode="bilinear",
        align_corners=False,
    )
    return native.squeeze(0) if squeeze else native


def crop_lon_wrap(field: torch.Tensor, lon_start: int, width: int) -> torch.Tensor:
    """Crop the last dimension with periodic longitude wrapping."""

    full_width = field.shape[-1]
    lon_start = int(lon_start) % full_width
    end = lon_start + width
    if end <= full_width:
        return field[..., lon_start:end]
    return torch.cat(
        (field[..., lon_start:], field[..., : end - full_width]),
        dim=-1,
    )


def crop_native_tile(
    field: torch.Tensor,
    *,
    lat_start: int,
    lon_start: int,
    height: int,
    width: int,
) -> torch.Tensor:
    tile = field[..., int(lat_start) : int(lat_start) + height, :]
    return crop_lon_wrap(tile, lon_start, width)


def load_static_maps(root: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    root = Path(root)
    land = torch.from_numpy(
        np.load(root / "land_sea.npy")
    ).to(torch.float32)
    orography = torch.from_numpy(
        np.load(root / "orography.npy")
    ).to(torch.float32)
    if tuple(land.shape) != NATIVE_SHAPE or tuple(orography.shape) != NATIVE_SHAPE:
        raise ValueError("Static land/orography maps must have shape (721, 1440).")
    return land, orography


def build_v4_static_features(
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    *,
    geographic_weights: torch.Tensor,
    land_sea: torch.Tensor,
    orography: torch.Tensor,
    zenith: torch.Tensor,
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
            zenith.to(torch.float32),
        ),
        dim=0,
    )
    if extra.shape[-2:] != base.shape[-2:]:
        raise ValueError("Extra static maps do not match the coordinate grid.")
    return torch.cat((base, extra), dim=0)


class CoarseGlobalBranch(nn.Module):
    """2° full-Earth residual, zero-initialized so training starts at raw GFS."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        weather_channels: int = 4,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(hidden_channels, weather_channels, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(features))


class LeadAwareGatedResidualCNNV4(nn.Module):
    """v3 local residual CNN plus a 2° global branch and land/orography/zenith."""

    def __init__(
        self,
        *,
        hidden_channels: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 4, 2),
        dropout: float = 0.05,
        initial_gate: float = 0.02,
        static_channels: int = V4_STATIC_CHANNELS,
        context_channels: int = CONTEXT_FEATURES,
    ) -> None:
        super().__init__()
        self.static_channels = static_channels
        self.local = LeadAwareGatedResidualCNN(
            weather_channels=4,
            static_channels=static_channels,
            context_channels=context_channels,
            hidden_channels=hidden_channels,
            dilations=dilations,
            dropout=dropout,
            initial_gate=initial_gate,
        )
        self.coarse = CoarseGlobalBranch(
            in_channels=4 + static_channels,
            hidden_channels=hidden_channels,
        )

    def forward(
        self,
        weather: torch.Tensor,
        context: torch.Tensor,
        static_features: torch.Tensor,
        zonal_mean: torch.Tensor,
        lat_starts: torch.Tensor,
        coarse_input: torch.Tensor,
        lon_starts: torch.Tensor | None = None,
    ) -> GatedResidualOutput:
        local = self.local(
            weather,
            context,
            static_features,
            zonal_mean=zonal_mean,
            lat_starts=lat_starts,
        )
        coarse_residual = self.coarse(coarse_input)
        coarse_native = upsample_native(coarse_residual)
        tiles = []
        height = weather.shape[-2]
        width = weather.shape[-1]
        if lon_starts is None:
            lon_starts = weather.new_zeros(weather.shape[0], dtype=torch.long)
        for index in range(weather.shape[0]):
            tiles.append(
                crop_native_tile(
                    coarse_native[index],
                    lat_start=int(lat_starts[index].item()),
                    lon_start=int(lon_starts[index].item()),
                    height=height,
                    width=width,
                )
            )
        coarse_tile = torch.stack(tiles, dim=0)
        return GatedResidualOutput(
            correction=local.correction + coarse_tile,
            gate=local.gate,
            ungated_residual=local.ungated_residual,
            zonal_gate=local.zonal_gate,
            horizon_mix=local.horizon_mix,
        )


def valid_time_from_cycle(cycle_time: datetime, lead_hour: float) -> datetime:
    from datetime import timedelta

    utc = (
        cycle_time.replace(tzinfo=timezone.utc)
        if cycle_time.tzinfo is None
        else cycle_time.astimezone(timezone.utc)
    )
    return utc + timedelta(hours=float(lead_hour))


# Re-export for callers that already import context helpers from the v3 module.
__all__ = [
    "COARSE_HEIGHT",
    "COARSE_WIDTH",
    "LeadAwareGatedResidualCNNV4",
    "V4_STATIC_CHANNELS",
    "build_v4_static_features",
    "context_from_cycle",
    "cosine_solar_zenith",
    "crop_native_tile",
    "downsample_2deg",
    "load_static_maps",
    "upsample_native",
    "valid_time_from_cycle",
]
