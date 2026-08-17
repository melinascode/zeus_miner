from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from zeus_ml.datasets.lead_aware_patch_dataset import (
    ChannelStatistics,
    LeadAwarePatchDataset,
    sample_patch_origins,
)
from zeus_ml.evaluate.evaluate_lead_aware_residual_cnn import (
    StreamingValidatorMetrics,
)
from zeus_ml.losses.validator_aware_residual import (
    ValidatorAwareResidualLoss,
)
from zeus_ml.models.lead_aware_residual_cnn import (
    LeadAwareGatedResidualCNN,
    build_static_features,
    build_temporal_context,
    crop_zonal_profile,
    horizon_mix_from_context,
)


def test_temporal_context_is_bounded_and_lead_aware() -> None:
    context = build_temporal_context(
        torch.tensor([0.0, 180.0, 360.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0]),
    )
    assert context.shape == (3, 7)
    assert torch.isfinite(context).all()
    assert context[0, 0] == 0.0
    assert context[-1, 0] == 1.0
    assert not torch.equal(context[0], context[1])


def test_model_starts_close_to_raw_forecast() -> None:
    model = LeadAwareGatedResidualCNN(
        hidden_channels=16,
        dilations=(1, 2, 4),
        dropout=0.0,
        initial_gate=0.02,
    )
    weather = torch.randn(2, 4, 16, 24)
    context = build_temporal_context(
        torch.zeros(2),
        torch.zeros(2),
        torch.ones(2),
    )
    static = build_static_features(
        torch.linspace(-20.0, 20.0, 16),
        torch.linspace(-30.0, 30.0, 24),
    )
    output = model(weather, context, static)
    assert output.correction.shape == weather.shape
    assert output.gate.shape == weather.shape
    assert output.ungated_residual.shape == weather.shape
    assert output.zonal_gate.shape[:2] == (2, 4)
    assert float(output.gate.mean().detach()) == pytest.approx(0.02, abs=1e-4)
    assert float(output.horizon_mix.mean().detach()) < 0.05
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000


def test_validator_aware_loss_rewards_perfect_correction() -> None:
    criterion = ValidatorAwareResidualLoss(
        correction_weight=0.0,
        gate_weight=0.0,
    )
    raw = torch.zeros(2, 4, 6, 8)
    truth = torch.ones_like(raw)
    weights = torch.ones(2, 1, 6, 8)
    scales = torch.ones(1, 4, 1, 1)
    gate = torch.ones_like(raw)
    perfect = criterion(
        raw_forecast=raw,
        truth=truth,
        correction=torch.ones_like(raw),
        gate=gate,
        metric_weights=weights,
        residual_scales=scales,
    )
    harmful = criterion(
        raw_forecast=raw,
        truth=truth,
        correction=-torch.ones_like(raw),
        gate=gate,
        metric_weights=weights,
        residual_scales=scales,
    )
    assert perfect.corrected_combined_error < perfect.raw_combined_error
    assert perfect.no_regret_penalty == pytest.approx(0.0)
    assert harmful.no_regret_penalty > 0.0
    assert harmful.loss > perfect.loss


def test_model_and_validator_loss_support_backward() -> None:
    model = LeadAwareGatedResidualCNN(
        hidden_channels=8,
        dilations=(1, 2),
        dropout=0.0,
    )
    criterion = ValidatorAwareResidualLoss()
    raw = torch.randn(2, 4, 12, 16)
    truth = raw + 0.1 * torch.randn_like(raw)
    statistics = torch.ones(1, 4, 1, 1)
    output = model(
        raw,
        torch.randn(2, 7),
        build_static_features(
            torch.linspace(-20.0, 20.0, 12),
            torch.linspace(-30.0, 30.0, 16),
        ),
    )
    loss = criterion(
        raw_forecast=raw,
        truth=truth,
        correction=output.correction,
        gate=output.gate,
        metric_weights=torch.ones(2, 1, 12, 16),
        residual_scales=statistics,
    ).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_streaming_metrics_match_direct_weighted_formula() -> None:
    prediction = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0]]] * 4,
            [[[2.0, 3.0], [4.0, 5.0]]] * 4,
        ]
    )
    truth = torch.zeros_like(prediction)
    weights = torch.tensor([[0.5, 1.0], [1.5, 1.0]])
    accumulator = StreamingValidatorMetrics()
    for lead in range(prediction.shape[0]):
        accumulator.update(prediction[lead], truth[lead], weights)
    metrics = accumulator.finalize()
    error = prediction[:, 0]
    expected_rmse = torch.sqrt((error.square() * weights).mean())
    expected_mae = (error.abs() * weights).mean()
    assert metrics[0]["rmse"] == pytest.approx(float(expected_rmse))
    assert metrics[0]["mae"] == pytest.approx(float(expected_mae))


def test_patch_dataset_returns_context_and_normalized_fields(tmp_path) -> None:
    cycle = "20260107T000000Z"
    cycle_dir = tmp_path / cycle
    cycle_dir.mkdir()
    shape = (1, 1, 4, 4, 4)
    inputs = np.full(shape, 10.0, dtype=np.float16)
    residuals = np.full(shape, 2.0, dtype=np.float16)
    np.save(cycle_dir / "inputs.npy", inputs)
    np.save(cycle_dir / "residuals.npy", residuals)
    np.save(cycle_dir / "lead_hours.npy", np.array([12], dtype=np.int16))
    np.save(cycle_dir / "lat_starts.npy", np.array([[100]], dtype=np.int32))
    np.save(cycle_dir / "lon_starts.npy", np.array([[200]], dtype=np.int32))
    np.save(
        cycle_dir / "zonal_means.npy",
        np.full((1, 4, 721), 10.0, dtype=np.float16),
    )
    (cycle_dir / "metadata.json").write_text(
        json.dumps(
            {
                "variables": [
                    "2m_temperature",
                    "100m_u_component_of_wind",
                    "100m_v_component_of_wind",
                    "surface_solar_radiation_downwards",
                ],
                "full_shape": [721, 1440],
                "global_metric_weight_mean": 1.0,
            }
        )
    )
    statistics = ChannelStatistics(
        gfs_mean=(8.0, 8.0, 8.0, 8.0),
        gfs_std=(2.0, 2.0, 2.0, 2.0),
        residual_std=(4.0, 4.0, 4.0, 4.0),
    )
    dataset = LeadAwarePatchDataset(
        tmp_path,
        cycles=[cycle],
        statistics=statistics,
    )
    sample = dataset[0]
    assert sample["model_input"].shape == (4, 4, 4)
    assert torch.allclose(sample["model_input"], torch.ones(4, 4, 4))
    assert torch.allclose(
        sample["residual_target"],
        torch.full((4, 4, 4), 0.5),
    )
    assert sample["context"].shape == (7,)
    assert sample["static_features"].shape == (5, 4, 4)
    assert sample["metric_weights"].shape == (1, 4, 4)
    assert sample["zonal_mean"].shape == (4, 721)
    assert sample["lat_start"].ndim == 0


def test_long_horizon_heads_open_the_gate() -> None:
    model = LeadAwareGatedResidualCNN(
        hidden_channels=8,
        dilations=(1, 2),
        dropout=0.0,
        initial_gate=0.02,
    )
    weather = torch.zeros(2, 4, 8, 8)
    static = build_static_features(
        torch.linspace(-10.0, 10.0, 8),
        torch.linspace(-10.0, 10.0, 8),
    )
    short = model(
        weather,
        build_temporal_context(torch.zeros(2), torch.zeros(2), torch.ones(2)),
        static,
    )
    long = model(
        weather,
        build_temporal_context(
            torch.full((2,), 360.0),
            torch.zeros(2),
            torch.ones(2),
        ),
        static,
    )
    assert float(short.horizon_mix.mean().detach()) < 0.05
    assert float(long.horizon_mix.mean().detach()) > 0.95
    assert float(long.gate.mean().detach()) > float(short.gate.mean().detach())


def test_zonal_profile_is_cropped_to_patch_latitudes() -> None:
    profile = torch.arange(721, dtype=torch.float32).view(1, 1, 721).expand(1, 4, 721).clone()
    cropped = crop_zonal_profile(
        profile,
        torch.tensor([100]),
        16,
    )
    assert cropped.shape == (1, 4, 16)
    assert float(cropped[0, 0, 0]) == 100.0
    assert float(cropped[0, 0, -1]) == 115.0


def test_patch_origins_cover_europe_and_germany() -> None:
    lat_starts, lon_starts = sample_patch_origins(
        np.random.default_rng(7),
        n_leads=3,
        patches_per_lead=4,
        patch_size=128,
        include_germany=True,
    )
    assert lat_starts.shape == (3, 4)
    europe_lat = -90.0 + (lat_starts[:, 0] + 64) * 0.25
    europe_lon = -180.0 + (lon_starts[:, 0] + 64) * 0.25
    germany_lat = -90.0 + (lat_starts[:, 1] + 64) * 0.25
    germany_lon = -180.0 + (lon_starts[:, 1] + 64) * 0.25
    assert np.all((europe_lat >= 34.0) & (europe_lat <= 72.0))
    assert np.all((europe_lon >= -25.0) & (europe_lon <= 45.0))
    assert np.all((germany_lat >= 47.0) & (germany_lat <= 56.0))
    assert np.all((germany_lon >= 6.0) & (germany_lon <= 15.0))


def test_horizon_mix_is_near_zero_then_one() -> None:
    mix = horizon_mix_from_context(
        build_temporal_context(
            torch.tensor([0.0, 48.0, 360.0]),
            torch.zeros(3),
            torch.ones(3),
        )
    )
    assert float(mix[0]) < 0.03
    assert float(mix[1]) == pytest.approx(0.5, abs=0.05)
    assert float(mix[2]) > 0.99
