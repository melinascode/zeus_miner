from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


WANDB_ENTITY = "castelcasey79-marzoni-s-brick-oven-brewing"
WANDB_PROJECT = "zeus-forecast-evaluation"
WANDB_GROUP = "validator-faithful-backtest-v1"
WANDB_BENCHMARK_GROUP = "validator-faithful-benchmark-v1"
REQUIRED_ENVIRONMENT = Path("/root/miniconda3/envs/zeus-eval")


def log_evaluation_to_wandb(
    result: dict[str, Any],
    result_path: str | Path,
    *,
    mode: str = "online",
) -> str:
    """Log one evaluation from the dedicated evaluator environment."""

    current_environment = Path(sys.prefix).resolve()
    if current_environment != REQUIRED_ENVIRONMENT.resolve():
        raise RuntimeError(
            "W&B logging must run from "
            f"{REQUIRED_ENVIRONMENT}; current environment is "
            f"{current_environment}."
        )

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is unavailable in the zeus-eval environment."
        ) from exc

    result_path = Path(result_path).resolve()
    case = result["case"]
    has_calibrated = "calibrated_gfs" in result.get("metrics", {})
    group = WANDB_BENCHMARK_GROUP if has_calibrated else WANDB_GROUP
    prefix = "vfb-b1" if has_calibrated else "vfb-v1"
    job_type = (
        "three-model-benchmark" if has_calibrated else "offline-evaluation"
    )
    run_name = (
        f"{prefix}__{case['cycle_key']}__{case['variable']}"
        f"__h0-{case['horizon_hours']}__{case['evaluation_id']}"
    )
    config = {
        **case,
        "evaluation_schema_version": result["schema_version"],
        "evaluator_git_revision": result["evaluator_git_revision"],
        "evaluator_repository_dirty": result["evaluator_repository_dirty"],
        "evaluator_source_sha256": result["evaluator_source_sha256"],
        "validator_metric": result["validator_behavior"]["metric"],
        "aggregation_dimensions": result["validator_behavior"][
            "aggregation_dimensions"
        ],
        "region_regime": result["validator_behavior"]["region_regime"],
        "variable_weight": result["validator_behavior"]["variable_weight"],
        "horizon_weight": result["validator_behavior"]["horizon_weight"],
        "challenge_weight": result["validator_behavior"]["challenge_weight"],
        "manifest_sha256": result["provenance"]["forecast"][
            "manifest_sha256"
        ],
        "manifest_authenticated": result["provenance"]["forecast"][
            "manifest_authenticated"
        ],
        "commitment_authenticated": result["provenance"]["forecast"][
            "commitment_authenticated"
        ],
        "payload_sha256": result["provenance"]["forecast"][
            "payload_sha256"
        ],
        "truth_file_sha256": result["provenance"]["truth"]["file_sha256"],
    }
    if result["provenance"].get("calibration"):
        config["coefficients_sha256"] = result["provenance"]["calibration"][
            "coefficients_sha256"
        ]
        config["selection_sha256"] = result["provenance"]["calibration"][
            "selection_sha256"
        ]

    with wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=group,
        name=run_name,
        job_type=job_type,
        config=config,
        mode=mode,
        dir=str(result_path.parent),
    ) as run:
        summary: dict[str, Any] = {
            "case/status": case.get("status", "complete"),
        }
        for model_name in ("persistence", "raw_gfs", "calibrated_gfs"):
            model_metrics = result["metrics"].get(model_name)
            if not model_metrics:
                continue
            summary[f"{model_name}/validator_regional_weighted_rmse"] = (
                model_metrics["rmse"]
            )
            summary[f"{model_name}/validator_regional_weighted_mae"] = (
                model_metrics["mae"]
            )
            summary[f"{model_name}/validator_combined_error"] = (
                model_metrics["combined_error"]
            )

        comparison = result["metrics"]["comparison"]
        for key, value in comparison.items():
            summary[f"comparison/{key}"] = value

        for key, value in summary.items():
            run.summary[key] = value

        lead_rows = result["lead_metrics_diagnostic"]
        if lead_rows:
            columns = [
                "lead_hour",
                "valid_time_utc",
                "raw_gfs_diagnostic_rmse",
                "raw_gfs_diagnostic_mae",
                "persistence_diagnostic_rmse",
                "persistence_diagnostic_mae",
            ]
            if has_calibrated:
                columns.extend(
                    [
                        "calibrated_gfs_diagnostic_rmse",
                        "calibrated_gfs_diagnostic_mae",
                    ]
                )
            data = []
            for row in lead_rows:
                values = [
                    row["lead_hour"],
                    row["valid_time_utc"],
                    row["raw_gfs"]["regional_weighted_rmse"],
                    row["raw_gfs"]["regional_weighted_mae"],
                    row["persistence"]["regional_weighted_rmse"],
                    row["persistence"]["regional_weighted_mae"],
                ]
                if has_calibrated:
                    calibrated = row["calibrated_gfs"]
                    values.extend(
                        [
                            calibrated["regional_weighted_rmse"],
                            calibrated["regional_weighted_mae"],
                        ]
                    )
                data.append(values)
            run.log({"diagnostic/lead_metrics": wandb.Table(columns=columns, data=data)})

        artifact = wandb.Artifact(
            name=f"{run_name}-results",
            type="evaluation-results",
            metadata={
                "manifest_sha256": result["provenance"]["forecast"][
                    "manifest_sha256"
                ],
                "payload_sha256": result["provenance"]["forecast"][
                    "payload_sha256"
                ],
            },
        )
        artifact.add_file(str(result_path), name="evaluation.json")
        run.log_artifact(artifact)
        return run.url or run.id
