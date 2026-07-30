from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.calibrated_gfs import apply_calibrated_gfs
from evaluation.calibration import (
    BiasAccumulator,
    FrozenCalibration,
    assert_no_test_cycle_in_fit,
)
from evaluation.runner import evaluate_case
from evaluation.scoring import ValidatorFaithfulScorer
from evaluation.selection import (
    content_sha256,
    load_selection,
    record_cycle_status,
    validate_selection,
)
from evaluation.truth import LoadedTruth
from forecast.forecast_store import ForecastStore
from evaluation.artifacts import EXPECTED_GFS_MODEL, ForecastArtifactReader
from zeus.utils.compression import compress_prediction


AFTER_GERMANY_CUTOFF = datetime(2026, 7, 1, tzinfo=timezone.utc)
SELECTION_PATH = Path(
    "data/evaluation/plans/benchmark_v1_selection.json"
)


class SmallGridScorer(ValidatorFaithfulScorer):
    def __init__(self, shape: tuple[int, int]) -> None:
        super().__init__(spatial_shape=shape)
        self._small_latitude = torch.linspace(0.5, 1.0, shape[0])
        self._small_geographic = torch.ones(shape)

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


def test_locked_selection_loads_and_rejects_mutation() -> None:
    selection = load_selection(SELECTION_PATH)
    assert len(selection["test_cycles"]) == 30
    assert selection["content_sha256"] == content_sha256(selection)
    assert selection["overlap_proof"]["truth_gap_hours"] == 168

    mutated = copy.deepcopy(selection)
    mutated["test_cycles"][0]["cycle"] = "20250422T120000Z"
    assert content_sha256(mutated) != selection["content_sha256"]
    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        load_selection_from_payload(mutated)


def load_selection_from_payload(payload: dict) -> dict:
    """Helper that mirrors file load hash checking for in-memory payloads."""
    digest = content_sha256(payload)
    recorded = payload.get("content_sha256")
    if recorded != digest:
        raise ValueError(
            f"Selection content_sha256 mismatch for memory: "
            f"recorded={recorded} computed={digest}"
        )
    validate_selection(payload)
    return payload


def test_selection_rejects_calib_test_overlap() -> None:
    selection = load_selection(SELECTION_PATH)
    bad = copy.deepcopy(selection)
    del bad["content_sha256"]
    bad["calibration"]["truth_end"] = selection["test_cycles"][0]["cycle"]
    with pytest.raises(ValueError, match="overlap"):
        validate_selection(bad)


def test_assert_no_test_cycle_in_fit() -> None:
    with pytest.raises(ValueError, match="leakage"):
        assert_no_test_cycle_in_fit(
            ["20250422T180000Z"],
            ["20250422T180000Z", "20250508T000000Z"],
        )


def test_bias_accumulator_and_frozen_apply(tmp_path: Path) -> None:
    latitudes = 2
    longitudes = 3
    leads = 361
    weights_path = tmp_path / "lat.npy"
    np.save(weights_path, np.array([0.5, 1.5], dtype=np.float64))

    accumulator = BiasAccumulator(
        latitude_weights_path=weights_path,
        n_leads=leads,
    )
    # Provide one sample per synoptic hour for every variable.
    for hour in (0, 6, 12, 18):
        cycle = datetime(2025, 2, 1, hour, tzinfo=timezone.utc)
        for variable in (
            "2m_temperature",
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "surface_solar_radiation_downwards",
        ):
            forecast = np.zeros(
                (leads, latitudes, longitudes),
                dtype=np.float32,
            )
            truth = np.full(
                (leads, latitudes, longitudes),
                2.0,
                dtype=np.float32,
            )
            if variable == "surface_solar_radiation_downwards":
                # Raw over-forecast so bias is negative and clip is exercised.
                forecast[:] = 5.0
                truth[:] = 1.0
            accumulator.update(
                variable=variable,
                cycle_time=cycle,
                forecast=forecast,
                truth=truth,
            )

    frozen = accumulator.freeze(selection_sha256="a" * 64)
    path = frozen.write(tmp_path / "coefficients.json")
    reloaded = FrozenCalibration.load(
        path,
        expected_selection_sha256="a" * 64,
    )
    assert reloaded.coefficients_sha256 == frozen.coefficients_sha256

    raw = torch.full((49, latitudes, longitudes), 5.0)
    calibrated = apply_calibrated_gfs(
        raw,
        variable="surface_solar_radiation_downwards",
        cycle_time=datetime(2025, 2, 1, 0, tzinfo=timezone.utc),
        coefficients=reloaded,
    )
    assert torch.all(calibrated >= 0)
    # bias = 1 - 5 = -4 ⇒ 5 + (-4) = 1
    torch.testing.assert_close(
        calibrated,
        torch.ones_like(calibrated),
    )


def test_frozen_coefficients_reject_wrong_selection(tmp_path: Path) -> None:
    weights_path = tmp_path / "lat.npy"
    np.save(weights_path, np.ones(2))
    accumulator = BiasAccumulator(
        latitude_weights_path=weights_path,
        n_leads=361,
    )
    for hour in (0, 6, 12, 18):
        cycle = datetime(2025, 2, 1, hour, tzinfo=timezone.utc)
        for variable in (
            "2m_temperature",
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "surface_solar_radiation_downwards",
        ):
            forecast = np.zeros((361, 2, 3), dtype=np.float32)
            truth = np.ones((361, 2, 3), dtype=np.float32)
            accumulator.update(
                variable=variable,
                cycle_time=cycle,
                forecast=forecast,
                truth=truth,
            )
    frozen = accumulator.freeze(selection_sha256="a" * 64)
    path = frozen.write(tmp_path / "coefficients.json")
    with pytest.raises(ValueError, match="different selection"):
        FrozenCalibration.load(path, expected_selection_sha256="b" * 64)


def test_evaluate_case_three_models_share_truth(tmp_path: Path) -> None:
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

    weights_path = tmp_path / "lat.npy"
    np.save(weights_path, np.array([1.0, 1.0], dtype=np.float64))
    accumulator = BiasAccumulator(
        latitude_weights_path=weights_path,
        n_leads=361,
    )
    for hour in (0, 6, 12, 18):
        for variable in (
            "2m_temperature",
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "surface_solar_radiation_downwards",
        ):
            forecast = np.zeros((361, 2, 3), dtype=np.float32)
            truth_arr = np.ones((361, 2, 3), dtype=np.float32)
            accumulator.update(
                variable=variable,
                cycle_time=datetime(2025, 2, 1, hour, tzinfo=timezone.utc),
                forecast=forecast,
                truth=truth_arr,
            )
    frozen = accumulator.freeze(selection_sha256="c" * 64)

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
        include_lead_diagnostics=False,
        calibration=frozen,
        selection_sha256="c" * 64,
    )
    assert result["schema_version"] == 2
    assert result["case"]["models"] == [
        "persistence",
        "raw_gfs",
        "calibrated_gfs",
    ]
    assert "calibrated_gfs" in result["metrics"]
    assert result["case"]["truth_shared_between_models"] is True
    with pytest.raises(ValueError, match="do not match the locked"):
        evaluate_case(
            artifact,
            truth,
            scorer=SmallGridScorer((2, 3)),
            calibration=frozen,
            selection_sha256="d" * 64,
        )


def test_registry_records_failed_without_changing_selection(
    tmp_path: Path,
) -> None:
    selection = load_selection(SELECTION_PATH)
    registry = {
        "schema_version": 1,
        "plan_id": "benchmark_v1",
        "selection_sha256": selection["content_sha256"],
        "cycles": {
            item["cycle"]: {
                "status": "pending",
                "failure_reason": None,
                "updated_at_utc": None,
            }
            for item in selection["test_cycles"]
        },
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    before = SELECTION_PATH.read_bytes()
    record_cycle_status(
        registry_path,
        selection["test_cycles"][0]["cycle"],
        "failed",
        failure_reason="missing ERA5",
        selection_sha256=selection["content_sha256"],
    )
    after = SELECTION_PATH.read_bytes()
    assert before == after
    updated = json.loads(registry_path.read_text())
    assert (
        updated["cycles"][selection["test_cycles"][0]["cycle"]]["status"]
        == "failed"
    )


def test_refusing_overwrite_different_coefficients(tmp_path: Path) -> None:
    weights_path = tmp_path / "lat.npy"
    np.save(weights_path, np.ones(2))

    def fit(tag: str) -> FrozenCalibration:
        accumulator = BiasAccumulator(
            latitude_weights_path=weights_path,
            n_leads=361,
        )
        for hour in (0, 6, 12, 18):
            for variable in (
                "2m_temperature",
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
                "surface_solar_radiation_downwards",
            ):
                forecast = np.zeros((361, 2, 3), dtype=np.float32)
                truth = np.full((361, 2, 3), float(ord(tag[0])), dtype=np.float32)
                accumulator.update(
                    variable=variable,
                    cycle_time=datetime(
                        2025, 2, 1, hour, tzinfo=timezone.utc
                    ),
                    forecast=forecast,
                    truth=truth,
                )
        return accumulator.freeze(selection_sha256="e" * 64)

    first = fit("a")
    path = first.write(tmp_path / "coefficients.json")
    second = fit("b")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        second.write(path)
