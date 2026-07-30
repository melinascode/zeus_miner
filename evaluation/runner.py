from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.artifacts import EXPECTED_GFS_MODEL, LoadedForecastArtifact
from evaluation.baselines import persistence_from_initial_field
from evaluation.calibrated_gfs import apply_calibrated_gfs
from evaluation.calibration import FrozenCalibration
from evaluation.scoring import ValidatorFaithfulScorer
from evaluation.truth import LoadedTruth
from zeus.validator.constants import CHALLENGE_REGISTRY


EVALUATION_SCHEMA_VERSION = 1
BENCHMARK_SCHEMA_VERSION = 2


def evaluate_case(
    artifact: LoadedForecastArtifact,
    truth: LoadedTruth,
    *,
    scorer: ValidatorFaithfulScorer | None = None,
    include_lead_diagnostics: bool = True,
    calibration: FrozenCalibration | None = None,
    selection_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate persistence, raw GFS, and optionally calibrated GFS together."""

    _validate_pair(artifact, truth)
    if calibration is not None:
        if selection_sha256 is None:
            raise ValueError(
                "selection_sha256 is required when applying calibrated GFS."
            )
        if calibration.selection_sha256 != selection_sha256:
            raise ValueError(
                "Frozen calibration coefficients do not match the locked "
                "selection hash."
            )
    scorer = scorer or ValidatorFaithfulScorer()
    challenge_spec = CHALLENGE_REGISTRY[artifact.state_key]
    repository = _repository_provenance()

    raw_gfs_score = scorer.score(
        truth.tensor,
        artifact.tensor,
        cycle_time=artifact.cycle_time,
    )
    persistence = persistence_from_initial_field(artifact.tensor)
    persistence_score = scorer.score(
        truth.tensor,
        persistence,
        cycle_time=artifact.cycle_time,
    )

    calibrated = None
    calibrated_score = None
    if calibration is not None:
        calibrated = apply_calibrated_gfs(
            artifact.tensor,
            variable=artifact.variable,
            cycle_time=artifact.cycle_time,
            coefficients=calibration,
        )
        calibrated_score = scorer.score(
            truth.tensor,
            calibrated,
            cycle_time=artifact.cycle_time,
        )

    delta = (
        raw_gfs_score.combined_error
        - persistence_score.combined_error
    )
    skill = (
        1.0
        - raw_gfs_score.combined_error
        / persistence_score.combined_error
        if persistence_score.combined_error != 0.0
        else None
    )

    lead_metrics: list[dict[str, Any]] = []
    if include_lead_diagnostics:
        raw_rows = scorer.per_lead_diagnostics(
            truth.tensor,
            artifact.tensor,
            cycle_time=artifact.cycle_time,
        )
        persistence_rows = scorer.per_lead_diagnostics(
            truth.tensor,
            persistence,
            cycle_time=artifact.cycle_time,
        )
        calibrated_rows = (
            scorer.per_lead_diagnostics(
                truth.tensor,
                calibrated,
                cycle_time=artifact.cycle_time,
            )
            if calibrated is not None
            else None
        )
        for index, (valid_time, raw_row, persistence_row) in enumerate(
            zip(
                artifact.valid_times,
                raw_rows,
                persistence_rows,
                strict=True,
            )
        ):
            row = {
                "lead_hour": raw_row["lead_hour"],
                "valid_time_utc": valid_time.isoformat(),
                "raw_gfs": {
                    key: value
                    for key, value in raw_row.items()
                    if key != "lead_hour"
                },
                "persistence": {
                    key: value
                    for key, value in persistence_row.items()
                    if key != "lead_hour"
                },
            }
            if calibrated_rows is not None:
                calibrated_row = calibrated_rows[index]
                row["calibrated_gfs"] = {
                    key: value
                    for key, value in calibrated_row.items()
                    if key != "lead_hour"
                }
            lead_metrics.append(row)

    metrics: dict[str, Any] = {
        "raw_gfs": raw_gfs_score.as_dict(),
        "persistence": persistence_score.as_dict(),
        "comparison": {
            "combined_error_delta_vs_persistence": delta,
            "combined_error_skill_vs_persistence": skill,
            "raw_gfs_wins": delta < 0.0,
        },
    }
    if calibrated_score is not None and calibration is not None:
        cal_vs_persistence = (
            calibrated_score.combined_error
            - persistence_score.combined_error
        )
        cal_vs_raw = (
            calibrated_score.combined_error
            - raw_gfs_score.combined_error
        )
        metrics["calibrated_gfs"] = calibrated_score.as_dict()
        metrics["comparison"].update(
            {
                "combined_error_delta_vs_raw_gfs": cal_vs_raw,
                "calibrated_gfs_delta_vs_persistence": cal_vs_persistence,
                "calibrated_gfs_delta_vs_raw_gfs": cal_vs_raw,
                "calibrated_gfs_wins_vs_persistence": (
                    cal_vs_persistence < 0.0
                ),
                "calibrated_gfs_wins_vs_raw_gfs": cal_vs_raw < 0.0,
            }
        )

    result = {
        "schema_version": (
            BENCHMARK_SCHEMA_VERSION
            if calibration is not None
            else EVALUATION_SCHEMA_VERSION
        ),
        "evaluator_git_revision": repository["git_revision"],
        "evaluator_repository_dirty": repository["dirty"],
        "evaluator_source_sha256": repository["source_sha256"],
        "case": {
            "cycle_key": artifact.cycle_time.strftime("%Y%m%dT%H%M%SZ"),
            "cycle_start_utc": artifact.cycle_time.isoformat(),
            "state_key": artifact.state_key,
            "variable": artifact.variable,
            "horizon_hours": artifact.horizon_hours,
            "requested_hours": artifact.horizon_hours + 1,
            "step_size_hours": 1,
            "grid_shape": list(artifact.tensor.shape[1:]),
            "target_unit": truth.target_unit,
            "truth_shared_between_models": True,
            "persistence_source": "raw_gfs_h000",
            "future_truth_used_by_persistence": False,
            "models": (
                ["persistence", "raw_gfs", "calibrated_gfs"]
                if calibration is not None
                else ["persistence", "raw_gfs"]
            ),
            "status": "complete",
            "selection_sha256": selection_sha256,
            "coefficients_sha256": (
                calibration.coefficients_sha256
                if calibration is not None
                else None
            ),
        },
        "validator_behavior": {
            "metric": (
                "mean_of_regional_latitude_weighted_rmse_and_mae"
            ),
            "aggregation_dimensions": [
                "time",
                "latitude",
                "longitude",
            ],
            "prediction_input_dtype": str(artifact.tensor.dtype),
            "scoring_dtype": "float32",
            "region_regime": raw_gfs_score.region_regime,
            "variable_weight": sum(
                spec.weight
                for spec in CHALLENGE_REGISTRY.values()
                if spec.variable == artifact.variable
            ),
            "horizon_weight": (
                challenge_spec.weight
                / sum(
                    spec.weight
                    for spec in CHALLENGE_REGISTRY.values()
                    if spec.variable == artifact.variable
                )
            ),
            "challenge_weight": challenge_spec.weight,
        },
        "metrics": metrics,
        "provenance": {
            "forecast": {
                "store_cycle_key": artifact.manifest["cycle_key"],
                "model": artifact.manifest.get("model"),
                "fallback": artifact.manifest.get("fallback", False),
                "gfs_common_cycle_utc": artifact.manifest.get(
                    "gfs_common_cycle_utc"
                ),
                "gfs_source_offset_hours": artifact.manifest.get(
                    "gfs_source_offset_hours"
                ),
                "manifest_sha256": artifact.manifest_sha256,
                "manifest_authenticated": artifact.manifest_authenticated,
                "payload_sha256": artifact.payload_sha256,
                "commitment_hash": artifact.commitment_hash,
                "commitment_authenticated": (
                    artifact.commitment_authenticated
                ),
                "hotkey": artifact.hotkey,
                "artifact_metadata": artifact.artifact_metadata,
            },
            "truth": {
                "product": "reanalysis-era5-single-levels",
                "variable": truth.variable,
                "source_unit": truth.source_unit,
                "target_unit": truth.target_unit,
                "files": list(truth.source_files),
                "file_sha256": list(truth.source_sha256),
                "coordinate_sha256": {
                    "latitude": _array_sha256(truth.latitudes),
                    "longitude": _array_sha256(truth.longitudes),
                },
            },
            "calibration": (
                {
                    "selection_sha256": calibration.selection_sha256,
                    "coefficients_sha256": (
                        calibration.coefficients_sha256
                    ),
                    "formula": calibration.payload["formula"],
                }
                if calibration is not None
                else None
            ),
        },
        "lead_metrics_diagnostic": lead_metrics,
    }
    identity = {
        "cycle_key": result["case"]["cycle_key"],
        "state_key": artifact.state_key,
        "payload_sha256": artifact.payload_sha256,
        "manifest_sha256": artifact.manifest_sha256,
        "truth_sha256": list(truth.source_sha256),
        "evaluator_source_sha256": repository["source_sha256"],
        "selection_sha256": selection_sha256,
        "coefficients_sha256": result["case"]["coefficients_sha256"],
    }
    result["case"]["evaluation_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return result


def write_evaluation_result(
    result: dict[str, Any],
    output_directory: str | Path = "data/evaluation",
) -> Path:
    """Atomically write deterministic JSON under the evaluation data root."""

    root = Path(output_directory)
    case = result["case"]
    destination = (
        root
        / case["cycle_key"]
        / case["state_key"]
        / case["evaluation_id"]
        / "evaluation.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        result,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if destination.is_file():
        if destination.read_bytes() == content:
            return destination
        raise FileExistsError(
            f"Refusing to overwrite different evaluation: {destination}"
        )
    temporary = destination.with_suffix(".json.tmp")
    try:
        with temporary.open("wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _validate_pair(
    artifact: LoadedForecastArtifact,
    truth: LoadedTruth,
) -> None:
    if artifact.manifest.get("model") != EXPECTED_GFS_MODEL:
        raise ValueError(
            f"Forecast model must be {EXPECTED_GFS_MODEL!r} for a raw-GFS "
            "comparison."
        )
    if not artifact.commitment_authenticated:
        raise ValueError(
            "Forecast commitment was not authenticated against a trusted hash."
        )
    if not artifact.manifest_authenticated:
        raise ValueError(
            "Forecast manifest was not authenticated against a trusted digest."
        )
    if artifact.variable != truth.variable:
        raise ValueError(
            f"Forecast variable {artifact.variable} does not match "
            f"truth variable {truth.variable}."
        )
    if artifact.cycle_time != truth.cycle_time:
        raise ValueError("Forecast and truth cycle times differ.")
    if artifact.horizon_hours != truth.horizon_hours:
        raise ValueError("Forecast and truth horizons differ.")
    if artifact.tensor.shape != truth.tensor.shape:
        raise ValueError(
            f"Forecast shape {tuple(artifact.tensor.shape)} does not "
            f"match truth shape {tuple(truth.tensor.shape)}."
        )
    if artifact.valid_times != truth.valid_times:
        raise ValueError("Forecast and truth valid times differ.")
    if artifact.latitudes.shape != truth.latitudes.shape or not np.allclose(
        artifact.latitudes,
        truth.latitudes,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Forecast and truth latitude coordinates differ.")
    if (
        artifact.longitudes.shape != truth.longitudes.shape
        or not np.allclose(
            artifact.longitudes,
            truth.longitudes,
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ValueError("Forecast and truth longitude coordinates differ.")


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous)).hexdigest()


def _repository_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    revision = _git_output(["git", "rev-parse", "HEAD"], root)
    status = _git_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        root,
    )

    source_paths = list((root / "evaluation").rglob("*.py"))
    source_paths.extend((root / "forecast").rglob("*.py"))
    source_paths.extend((root / "zeus").rglob("*.py"))
    source_paths.extend(
        root / relative
        for relative in (
            "tools/evaluate_forecast_bundle.py",
            "tools/run_evaluation_backtest.py",
            "tools/log_evaluation_wandb.py",
            "tools/run_calibration_fit.py",
            "tools/build_historical_gfs_bundle.py",
            "tools/fetch_era5_evaluation.py",
            "zeus/data/weights/latitude_weights_for_rmse.npy",
        )
        if (root / relative).exists()
    )
    digest = hashlib.sha256()
    for path in sorted(source_paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return {
        "git_revision": revision,
        "dirty": bool(status),
        "source_sha256": digest.hexdigest(),
    }


def _git_output(command: list[str], root: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None
