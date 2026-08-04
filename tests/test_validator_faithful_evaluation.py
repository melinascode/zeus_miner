from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from evaluation.artifacts import EXPECTED_GFS_MODEL, ForecastArtifactReader
from evaluation.backtest import (
    BacktestCase,
    SUPPORTED_HORIZONS,
    SUPPORTED_VARIABLES,
    summarize_results,
    validate_backtest_plan,
)
from evaluation.baselines import persistence_from_initial_field
from evaluation.runner import evaluate_case, write_evaluation_result
from evaluation.scoring import ValidatorFaithfulScorer
from evaluation.truth import Era5TruthLoader, LoadedTruth
from forecast.forecast_store import ForecastStore
from zeus.utils.compression import compress_prediction
from zeus.validator.metrics import custom_mae, custom_rmse


AFTER_GERMANY_CUTOFF = datetime(2026, 7, 1, tzinfo=timezone.utc)


class SmallGridScorer(ValidatorFaithfulScorer):
    def __init__(self, shape: tuple[int, int]) -> None:
        super().__init__(spatial_shape=shape)
        self._small_latitude = torch.linspace(0.5, 1.0, shape[0])
        self._small_geographic = torch.ones(shape)
        self._small_geographic[-1, -1] = 2.5

    def _latitude_weights(self, override):
        return (
            super()._latitude_weights(override)
            if override is not None
            else self._small_latitude
        )

    def _geographic_weights(self, cycle_time, override):
        return (
            super()._geographic_weights(cycle_time, override)
            if override is not None
            else self._small_geographic
        )


def test_scoring_matches_validator_functions_exactly() -> None:
    truth = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    prediction = truth + torch.tensor(
        [
            [[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]],
            [[1.0, 0.0, -0.5], [1.5, -1.5, 0.0]],
            [[0.25, -0.25, 0.75], [0.0, 1.0, -1.0]],
            [[2.0, -2.0, 1.0], [0.5, -0.5, 0.0]],
        ]
    )
    latitude = torch.tensor([0.25, 1.0])
    geographic = torch.tensor([[1.0, 1.5, 1.0], [1.0, 2.5, 1.0]])

    scorer = ValidatorFaithfulScorer(spatial_shape=(2, 3))
    result = scorer.score(
        truth,
        prediction.to(torch.float16),
        cycle_time=AFTER_GERMANY_CUTOFF,
        latitude_weights=latitude,
        geographic_weights=geographic,
    )
    _, expected_rmse = custom_rmse(
        truth,
        prediction.to(torch.float16).to(torch.float32),
        latitude,
        geographic,
    )
    _, expected_mae = custom_mae(
        truth,
        prediction.to(torch.float16).to(torch.float32),
        latitude,
        geographic,
    )

    assert result.rmse == expected_rmse
    assert result.mae == expected_mae
    assert result.combined_error == (expected_rmse + expected_mae) / 2
    assert result.shape_penalty is False
    assert result.region_regime == "europe_germany"


def test_region_regime_changes_at_exact_validator_cutoff() -> None:
    before = datetime(2026, 4, 10, 5, 59, 59, tzinfo=timezone.utc)
    cutoff = datetime(2026, 4, 10, 6, 0, 0, tzinfo=timezone.utc)
    assert ValidatorFaithfulScorer.region_regime(before) == "europe_only"
    assert ValidatorFaithfulScorer.region_regime(cutoff) == "europe_germany"


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        torch.zeros((3, 2, 3)),
        torch.full((4, 2, 3), float("nan")),
        torch.full((4, 2, 3), float("inf")),
    ],
)
def test_invalid_predictions_receive_validator_penalty(invalid) -> None:
    scorer = ValidatorFaithfulScorer(spatial_shape=(2, 3))
    result = scorer.score(
        torch.zeros((4, 2, 3)),
        invalid,
        cycle_time=AFTER_GERMANY_CUTOFF,
        latitude_weights=torch.ones(2),
        geographic_weights=torch.ones((2, 3)),
    )
    assert result.shape_penalty is True
    assert result.rmse == float("inf")
    assert result.mae == float("inf")
    assert result.combined_error == float("inf")


def test_persistence_repeats_only_candidate_h000() -> None:
    forecast = torch.arange(4 * 2 * 3, dtype=torch.float16).reshape(4, 2, 3)
    persistence = persistence_from_initial_field(forecast)
    assert persistence.dtype == torch.float16
    assert persistence.stride(0) == 0
    for hour in range(4):
        torch.testing.assert_close(persistence[hour], forecast[0])


def test_artifact_reader_verifies_and_decodes_float16(tmp_path: Path) -> None:
    store = ForecastStore(tmp_path / "store")
    cycle = AFTER_GERMANY_CUTOFF
    state_key = "2m_temperature@0_48"
    hotkey = "5TestHotkey"
    values = np.arange(49 * 2 * 3, dtype=np.float16).reshape(49, 2, 3)
    payload = compress_prediction(values)
    commitment = hashlib.sha256(
        payload + hotkey.encode("utf-8")
    ).hexdigest()

    with store.begin_bundle(cycle, (state_key,)) as writer:
        writer.write_artifact(
            state_key,
            payload,
            shape=values.shape,
            dtype="float16",
            variable="2m_temperature",
            requested_hours=49,
            commitment_hash=commitment,
            source_valid_time_utc=cycle.isoformat(),
        )
        writer.finalize(
            {
                "hotkey": hotkey,
                "model": EXPECTED_GFS_MODEL,
                "fallback": False,
                "gfs_common_cycle_utc": cycle.isoformat(),
                "gfs_source_offset_hours": 0,
            }
        )

    manifest_sha256 = hashlib.sha256(
        (store.bundle_path(cycle) / store.MANIFEST_FILENAME).read_bytes()
    ).hexdigest()
    loaded = ForecastArtifactReader(
        tmp_path / "store",
        spatial_shape=(2, 3),
        require_complete_bundle=False,
    ).read(
        cycle,
        "2m_temperature",
        48,
        expected_commitment_hash=commitment,
        expected_manifest_sha256=manifest_sha256,
    )
    np.testing.assert_array_equal(loaded.tensor.numpy(), values)
    assert loaded.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert loaded.commitment_hash == commitment
    assert loaded.commitment_authenticated is True
    assert loaded.manifest_authenticated is True
    assert len(loaded.valid_times) == 49
    assert loaded.valid_times[-1] == cycle.replace(hour=0) + (
        loaded.valid_times[-1] - loaded.valid_times[0]
    )


def test_artifact_reader_rejects_rounded_cycle_time(tmp_path: Path) -> None:
    ForecastStore(tmp_path / "store")
    reader = ForecastArtifactReader(
        tmp_path / "store",
        require_complete_bundle=False,
    )
    with pytest.raises(ValueError, match="exact UTC"):
        reader.parse_cycle("2026-07-01T03:00:00Z")


def test_truth_loader_rejects_missing_units_metadata() -> None:
    with pytest.raises(ValueError, match="no units metadata"):
        Era5TruthLoader._validate_source_unit("2m_temperature", None)


@pytest.mark.parametrize(
    (
        "variable",
        "short_code",
        "source_unit",
        "raw_value",
        "expected_value",
    ),
    [
        ("2m_temperature", "t2m", "K", 273.15, 273.15),
        (
            "surface_solar_radiation_downwards",
            "ssrd",
            "J m**-2",
            3600.0,
            1.0,
        ),
    ],
)
def test_truth_loader_aligns_time_grid_and_units(
    tmp_path: Path,
    variable: str,
    short_code: str,
    source_unit: str,
    raw_value: float,
    expected_value: float,
) -> None:
    cycle = AFTER_GERMANY_CUTOFF
    latitudes = np.array([-0.25, 0.25])
    longitudes = np.array([-0.5, 0.0, 0.5])
    times = np.arange(
        np.datetime64("2026-07-01T00"),
        np.datetime64("2026-07-03T01"),
        np.timedelta64(1, "h"),
    )
    values = np.full(
        (49, len(latitudes), len(longitudes)),
        raw_value,
        dtype=np.float32,
    )
    dataset = xr.Dataset(
        {
            short_code: (
                ("valid_time", "latitude", "longitude"),
                values,
            )
        },
        coords={
            "valid_time": times,
            "latitude": latitudes[::-1],
            "longitude": (longitudes + 360.0) % 360.0,
        },
    )
    dataset[short_code].values[:] = values[:, ::-1]
    dataset[short_code].attrs["units"] = source_unit
    path = tmp_path / f"{short_code}.nc"
    dataset.to_netcdf(path, engine="h5netcdf")

    truth = Era5TruthLoader(
        target_latitudes=latitudes,
        target_longitudes=longitudes,
    ).load(
        [path],
        variable=variable,
        cycle_time=cycle,
        horizon_hours=48,
    )
    assert truth.tensor.shape == (49, 2, 3)
    assert truth.tensor.dtype == torch.float32
    torch.testing.assert_close(
        truth.tensor,
        torch.full_like(truth.tensor, expected_value),
    )
    assert truth.valid_times[0] == cycle
    assert truth.valid_times[-1] == datetime(
        2026,
        7,
        3,
        tzinfo=timezone.utc,
    )


def test_pair_uses_identical_truth_and_writes_deterministically(
    tmp_path: Path,
) -> None:
    store = ForecastStore(tmp_path / "store")
    cycle = AFTER_GERMANY_CUTOFF
    state_key = "2m_temperature@0_48"
    hotkey = "5TestHotkey"
    values = np.arange(49 * 2 * 3, dtype=np.float16).reshape(49, 2, 3)
    payload = compress_prediction(values)
    commitment = hashlib.sha256(
        payload + hotkey.encode("utf-8")
    ).hexdigest()
    with store.begin_bundle(cycle, (state_key,)) as writer:
        writer.write_artifact(
            state_key,
            payload,
            shape=values.shape,
            dtype="float16",
            variable="2m_temperature",
            requested_hours=49,
            commitment_hash=commitment,
            source_valid_time_utc=cycle.isoformat(),
        )
        writer.finalize(
            {
                "hotkey": hotkey,
                "model": EXPECTED_GFS_MODEL,
                "fallback": False,
                "gfs_common_cycle_utc": cycle.isoformat(),
                "gfs_source_offset_hours": 0,
            }
        )

    manifest_sha256 = hashlib.sha256(
        (store.bundle_path(cycle) / store.MANIFEST_FILENAME).read_bytes()
    ).hexdigest()
    artifact = ForecastArtifactReader(
        tmp_path / "store",
        spatial_shape=(2, 3),
        require_complete_bundle=False,
    ).read(
        cycle,
        "2m_temperature",
        48,
        expected_commitment_hash=commitment,
        expected_manifest_sha256=manifest_sha256,
    )
    truth = LoadedTruth(
        cycle_time=cycle,
        variable="2m_temperature",
        horizon_hours=48,
        tensor=artifact.tensor.to(torch.float32) + 1.0,
        valid_times=artifact.valid_times,
        latitudes=artifact.latitudes.copy(),
        longitudes=artifact.longitudes.copy(),
        target_unit="K",
        source_unit="K",
        source_files=("truth.nc",),
        source_sha256=("abc123",),
    )
    result = evaluate_case(
        artifact,
        truth,
        scorer=SmallGridScorer((2, 3)),
        include_lead_diagnostics=True,
    )
    assert result["case"]["truth_shared_between_models"] is True
    assert result["case"]["future_truth_used_by_persistence"] is False
    assert len(result["lead_metrics_diagnostic"]) == 49

    first = write_evaluation_result(result, tmp_path / "evaluation")
    first_bytes = first.read_bytes()
    second = write_evaluation_result(result, tmp_path / "evaluation")
    assert second.read_bytes() == first_bytes
    changed = copy.deepcopy(result)
    changed["metrics"]["raw_gfs"]["rmse"] += 1.0
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_evaluation_result(changed, tmp_path / "evaluation")


def test_backtest_plan_enforces_full_independent_thirty_cycle_matrix() -> None:
    cases = []
    for cycle_number in range(30):
        cycle = AFTER_GERMANY_CUTOFF + timedelta(
            hours=366 * cycle_number
        )
        for variable in SUPPORTED_VARIABLES:
            for horizon in SUPPORTED_HORIZONS:
                cases.append(
                    BacktestCase(
                        cycle_time=cycle,
                        variable=variable,
                        horizon_hours=horizon,
                        truth_files=("truth.nc",),
                        commitment_hash="a" * 64,
                        manifest_sha256="c" * 64,
                    )
                )

    validated = validate_backtest_plan(
        cases,
        minimum_cycles=30,
        require_full_matrix=True,
        require_independent=True,
    )
    assert len(validated) == 30 * 4 * 2


def test_backtest_plan_rejects_overlapping_long_windows() -> None:
    cases = [
        BacktestCase(
            cycle_time=AFTER_GERMANY_CUTOFF,
            variable="2m_temperature",
            horizon_hours=360,
            truth_files=("truth.nc",),
            commitment_hash="a" * 64,
            manifest_sha256="c" * 64,
        ),
        BacktestCase(
            cycle_time=AFTER_GERMANY_CUTOFF + timedelta(hours=6),
            variable="2m_temperature",
            horizon_hours=360,
            truth_files=("truth.nc",),
            commitment_hash="b" * 64,
            manifest_sha256="d" * 64,
        ),
    ]
    with pytest.raises(ValueError, match="overlap"):
        validate_backtest_plan(cases, require_independent=True)


def test_backtest_summary_keeps_candidates_separate() -> None:
    def result(cycle_key: str, raw: float, persistence: float) -> dict:
        return {
            "case": {
                "cycle_key": cycle_key,
                "variable": "2m_temperature",
                "horizon_hours": 48,
            },
            "metrics": {
                "raw_gfs": {
                    "rmse": raw,
                    "mae": raw,
                    "combined_error": raw,
                },
                "persistence": {
                    "rmse": persistence,
                    "mae": persistence,
                    "combined_error": persistence,
                },
                "comparison": {"raw_gfs_wins": raw < persistence},
            },
        }

    summary = summarize_results(
        [
            result("20260701T000000Z", 1.0, 2.0),
            result("20260716T060000Z", 3.0, 4.0),
        ]
    )
    assert summary["unique_cycles"] == 2
    assert summary["raw_gfs_wins"] == 2
    assert len(summary["groups"]) == 2
