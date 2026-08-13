from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from zeus_ml.datasets.lead_aware_patch_dataset import (
    ChannelStatistics,
    LeadAwarePatchDataset,
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
    context = torch.randn(2, 7)
    static = build_static_features(
        torch.linspace(-20.0, 20.0, 16),
        torch.linspace(-30.0, 30.0, 24),
    )
    output = model(weather, context, static)
    assert output.correction.shape == weather.shape
    assert output.gate.shape == weather.shape
    assert output.ungated_residual.shape == weather.shape
    assert float(output.gate.mean().detach()) == pytest.approx(0.02, abs=1e-5)
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
