"""FiLM-conditioned ResUNet specialist for the Europe crop (208 x 368).

Corrects linearly interpolated ENS-mean/AIFS-Single forecasts over Europe,
scored by the official capacity scalars. Fully convolutional, so it trains on
random sub-crops and serves on the full domain. An untrained network outputs
exactly the linear interpolation (gate starts closed), mirroring the global
downscaler's no-regret design.

Self-contained on purpose (torch/numpy only): the GPU pod needs this file, the
training script and the crop dataset - nothing else from the repo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn

VARIABLES = ("2m_temperature", "100m_u_component_of_wind", "100m_v_component_of_wind")
N_VARS = 3
STEP_HOURS = 6
MAX_LEAD_HOURS = 360
# Validator variable weights (0.2, 0.3, 0.3, 0.2) with ssrd dropped, renormed.
VARIABLE_WEIGHTS = (0.25, 0.375, 0.375)

WEATHER_CHANNELS = 12  # state(3) + tendency(3) + lagged(3) + lagged tendency(3)
CLIM_CHANNELS = 3  # (interpolated - climatology) / state_std
ZENITH_CHANNELS = 2
STATIC_CHANNELS = 6  # land, orography, roughness, cos-lat, log temp/wind scalars
IN_CHANNELS = WEATHER_CHANNELS + CLIM_CHANNELS + ZENITH_CHANNELS + STATIC_CHANNELS
CONTEXT_FEATURES = 8


def bracket_for_lead(lead: int) -> tuple[int, int, float]:
    left = min(lead // STEP_HOURS, MAX_LEAD_HOURS // STEP_HOURS)
    right = min(left + 1, MAX_LEAD_HOURS // STEP_HOURS)
    fraction = (lead - left * STEP_HOURS) / STEP_HOURS if right != left else 0.0
    return left, right, float(fraction)


def cosine_solar_zenith(
    latitudes: torch.Tensor, longitudes: torch.Tensor, when: datetime
) -> torch.Tensor:
    """cos(solar zenith) on the lat x lon grid; standard declination approx."""
    doy = when.timetuple().tm_yday - 1 + when.hour / 24.0 + when.minute / 1440.0
    declination = -math.radians(23.44) * math.cos(
        2.0 * math.pi * (doy + 10.0) / 365.25
    )
    utc_fraction = (when.hour + when.minute / 60.0) / 24.0
    lat = torch.deg2rad(latitudes)[:, None]
    lon = torch.deg2rad(longitudes)[None, :]
    hour_angle = 2.0 * math.pi * utc_fraction + lon - math.pi
    return (
        torch.sin(lat) * math.sin(declination)
        + torch.cos(lat) * math.cos(declination) * torch.cos(hour_angle)
    ).to(torch.float32)


def harmonic_features(when: datetime) -> np.ndarray:
    """Matches tools/fit_era5_climatology.py exactly (5 annual x 5 diurnal)."""
    doy = when.timetuple().tm_yday - 1 + when.hour / 24.0
    annual = 2.0 * np.pi * doy / 365.25
    diurnal = 2.0 * np.pi * when.hour / 24.0
    a = np.array(
        [1.0, np.cos(annual), np.sin(annual), np.cos(2 * annual), np.sin(2 * annual)]
    )
    d = np.array(
        [
            1.0,
            np.cos(diurnal),
            np.sin(diurnal),
            np.cos(2 * diurnal),
            np.sin(2 * diurnal),
        ]
    )
    return np.outer(a, d).ravel()


def evaluate_climatology(coefficients: np.ndarray, when: datetime) -> np.ndarray:
    """(vars, 25, H, W) coefficients -> (vars, H, W) climatology."""
    features = harmonic_features(when).astype(coefficients.dtype)
    return np.tensordot(features, coefficients, axes=([0], [1]))


def build_context(cycle_time: datetime, lead: int) -> torch.Tensor:
    valid = cycle_time + timedelta(hours=lead)
    doy_phase = 2.0 * math.pi * (valid.timetuple().tm_yday - 1) / 365.25
    hour_phase = 2.0 * math.pi * valid.hour / 24.0
    _, _, fraction = bracket_for_lead(lead)
    return torch.tensor(
        [
            lead / MAX_LEAD_HOURS,
            math.sqrt(lead / MAX_LEAD_HOURS),
            math.cos(doy_phase),
            math.sin(doy_phase),
            math.cos(hour_phase),
            math.sin(hour_phase),
            fraction,
            1.0,
        ],
        dtype=torch.float32,
    )


class FiLMResBlock(nn.Module):
    def __init__(
        self, channels: int, context_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.film = nn.Linear(context_dim, 2 * channels)
        self.act = nn.SiLU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(context).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.drop(self.act(self.norm1(self.conv1(x)) * (1.0 + scale) + shift))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class Stage(nn.Module):
    def __init__(
        self,
        channels: int,
        context_dim: int,
        blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            FiLMResBlock(channels, context_dim, dropout) for _ in range(blocks)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, context)
        return x


@dataclass(frozen=True)
class EuropeOutput:
    correction: torch.Tensor  # (batch, 3, H, W), residual-std units
    gate: torch.Tensor  # (batch, 3, H, W) in (0, 1)


class EuropeResUNet(nn.Module):
    """3-level encoder/decoder; input H, W must be divisible by 8."""

    def __init__(
        self,
        base_channels: int = 64,
        context_dim: int = 64,
        in_channels: int = IN_CHANNELS,
        blocks_per_stage: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 6,
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(CONTEXT_FEATURES, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
        )
        self.stem = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.enc1 = Stage(c1, context_dim, blocks_per_stage, dropout)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = Stage(c2, context_dim, blocks_per_stage, dropout)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.enc3 = Stage(c3, context_dim, blocks_per_stage, dropout)
        self.down3 = nn.Conv2d(c3, c4, 3, stride=2, padding=1)
        self.bottleneck = Stage(c4, context_dim, blocks_per_stage, dropout)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = Stage(c3, context_dim, blocks_per_stage, dropout)
        self.fuse3 = nn.Conv2d(c3 * 2, c3, 1)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.fuse2 = nn.Conv2d(c2 * 2, c2, 1)
        self.dec2 = Stage(c2, context_dim, blocks_per_stage, dropout)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.fuse1 = nn.Conv2d(c1 * 2, c1, 1)
        self.dec1 = Stage(c1, context_dim, blocks_per_stage, dropout)
        self.head_correction = nn.Conv2d(c1, N_VARS, 3, padding=1)
        self.head_gate = nn.Conv2d(c1, N_VARS, 3, padding=1)
        # Start with no correction and a nearly closed gate: the untrained
        # model reproduces the linear interpolation it is asked to fix.
        nn.init.zeros_(self.head_correction.weight)
        nn.init.zeros_(self.head_correction.bias)
        nn.init.zeros_(self.head_gate.weight)
        nn.init.constant_(self.head_gate.bias, -2.0)

    def forward(
        self, features: torch.Tensor, context: torch.Tensor
    ) -> EuropeOutput:
        ctx = self.context_mlp(context)
        e1 = self.enc1(self.stem(features), ctx)
        e2 = self.enc2(self.down1(e1), ctx)
        e3 = self.enc3(self.down2(e2), ctx)
        b = self.bottleneck(self.down3(e3), ctx)
        d3 = self.dec3(self.fuse3(torch.cat([self.up3(b), e3], dim=1)), ctx)
        d2 = self.dec2(self.fuse2(torch.cat([self.up2(d3), e2], dim=1)), ctx)
        d1 = self.dec1(self.fuse1(torch.cat([self.up1(d2), e1], dim=1)), ctx)
        return EuropeOutput(
            correction=self.head_correction(d1),
            gate=torch.sigmoid(self.head_gate(d1)),
        )
