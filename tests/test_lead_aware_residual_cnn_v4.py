from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from zeus_ml.losses.validator_aware_residual import ValidatorAwareResidualLoss
from zeus_ml.models.lead_aware_residual_cnn import build_temporal_context
from zeus_ml.models.lead_aware_residual_cnn_v4 import (
    LeadAwareGatedResidualCNNV4,
    cosine_solar_zenith,
    crop_native_tile,
    downsample_2deg,
    upsample_native,
)


def test_lon_wrap_crop_is_periodic() -> None:
    field = torch.arange(1440, dtype=torch.float32).view(1, 1, 1440).expand(2, 8, 1440).clone()
    cropped = crop_native_tile(field, lat_start=0, lon_start=1400, height=8, width=80)
    assert cropped.shape == (2, 8, 80)
    assert float(cropped[0, 0, 0]) == 1400.0
    assert float(cropped[0, 0, 39]) == 1439.0
    assert float(cropped[0, 0, 40]) == 0.0


def test_2deg_roundtrip_preserves_shape() -> None:
    native = torch.randn(4, 721, 1440)
    coarse = downsample_2deg(native)
    assert coarse.shape == (4, 91, 180)
    assert upsample_native(coarse).shape == (4, 721, 1440)


def test_v4_forward_adds_coarse_and_local() -> None:
    model = LeadAwareGatedResidualCNNV4(
        hidden_channels=8,
        dilations=(1, 2),
        dropout=0.0,
        initial_gate=0.02,
    )
    weather = torch.zeros(1, 4, 512, 512)
    static = torch.zeros(1, 8, 512, 512)
    zonal = torch.zeros(1, 4, 721)
    coarse = torch.zeros(1, 12, 91, 180)
    context = build_temporal_context(torch.zeros(1), torch.zeros(1), torch.ones(1))
    output = model(
        weather,
        context,
        static,
        zonal,
        torch.zeros(1, dtype=torch.long),
        coarse,
        lon_starts=torch.zeros(1, dtype=torch.long),
    )
    assert output.correction.shape == weather.shape
    assert float(output.gate.mean().detach()) == pytest.approx(0.02, abs=1e-3)
    assert float(output.correction.abs().mean().detach()) < 0.02


def test_zenith_is_high_at_summer_noon_tropics() -> None:
    latitudes = torch.tensor([0.0])
    longitudes = torch.tensor([0.0])
    noon = datetime(2025, 6, 21, 12, 0, tzinfo=timezone.utc)
    midnight = datetime(2025, 6, 21, 0, 0, tzinfo=timezone.utc)
    assert float(cosine_solar_zenith(latitudes, longitudes, noon)) > 0.8
    assert float(cosine_solar_zenith(latitudes, longitudes, midnight)) < 0.0


def test_train_loss_accepts_temperature_heavy_weights() -> None:
    criterion = ValidatorAwareResidualLoss(
        variable_weights=(0.35, 0.30, 0.30, 0.05),
    )
    raw = torch.zeros(1, 4, 8, 8)
    truth = torch.ones(1, 4, 8, 8)
    correction = torch.zeros(1, 4, 8, 8)
    gate = torch.full((1, 4, 8, 8), 0.02)
    weights = torch.ones(1, 1, 8, 8)
    scales = torch.ones(4).view(1, 4, 1, 1)
    output = criterion(
        raw_forecast=raw,
        truth=truth,
        correction=correction,
        gate=gate,
        metric_weights=weights,
        residual_scales=scales,
    )
    assert torch.isfinite(output.loss)
    assert pytest.approx(0.35) == float(criterion.variable_weights[0])
