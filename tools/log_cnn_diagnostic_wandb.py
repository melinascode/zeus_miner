#!/usr/bin/env python3
"""Log the 360-hour CNN diagnostic result to Weights & Biases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.wandb_logging import (
    REQUIRED_ENVIRONMENT,
    WANDB_ENTITY,
    WANDB_PROJECT,
)


WANDB_GROUP = "lead-aware-cnn-v2-diagnostic"
VARIABLES = {
    "2m_temperature": ("temperature", "2m temperature", "K"),
    "100m_u_component_of_wind": ("u100", "100m u-wind", "m/s"),
    "100m_v_component_of_wind": ("v100", "100m v-wind", "m/s"),
    "surface_solar_radiation_downwards": ("ssrd", "SSRD", "W/m²"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--image",
        default=(
            "data/evaluation/results/lead_aware_residual_cnn_v2/"
            "diagnostic_20260709T000000Z/"
            "20260709T000000Z_per_lead_iwrmse_iwmae.png"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def skill(raw: float, corrected: float) -> float:
    return (raw - corrected) / raw


def main() -> int:
    args = parse_args()
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

    result_path = Path(args.result).resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cycle = result["cycles"][0]
    diagnostics = cycle["lead_diagnostics"]
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    run_name = f"cnn-v2__{cycle['cycle']}__h0-{result['horizon_hours']}"
    with wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=run_name,
        job_type="cnn-diagnostic",
        mode=args.mode,
        dir=str(result_path.parent),
        config={
            "cycle": cycle["cycle"],
            "horizon_hours": result["horizon_hours"],
            "lead_count": len(diagnostics),
            "region_regime": cycle["region_regime"],
            "split": result["split"],
            "benchmark_valid": result["benchmark_valid"],
            "model_type": result["model_type"],
            "checkpoint": result["checkpoint"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "evaluation_split_plan_sha256": result[
                "evaluation_split_plan_sha256"
            ],
        },
    ) as run:
        run.define_metric("lead_hour")
        for short, _label, _unit in VARIABLES.values():
            run.define_metric(f"{short}/*", step_metric="lead_hour")

        summary_rows = []
        for variable, (short, label, unit) in VARIABLES.items():
            metrics = cycle["variables"][variable]
            raw = metrics["raw_gfs"]
            cnn = metrics["cnn_corrected"]
            rmse_skill = skill(raw["rmse"], cnn["rmse"])
            mae_skill = skill(raw["mae"], cnn["mae"])
            combined_skill = metrics["combined_error_skill"]
            run.summary[f"{short}/raw_gfs/iwRMSE"] = raw["rmse"]
            run.summary[f"{short}/cnn/iwRMSE"] = cnn["rmse"]
            run.summary[f"{short}/raw_gfs/iwMAE"] = raw["mae"]
            run.summary[f"{short}/cnn/iwMAE"] = cnn["mae"]
            run.summary[f"{short}/raw_gfs/combined_error"] = raw[
                "combined_error"
            ]
            run.summary[f"{short}/cnn/combined_error"] = cnn["combined_error"]
            run.summary[f"{short}/iwRMSE_skill"] = rmse_skill
            run.summary[f"{short}/iwMAE_skill"] = mae_skill
            run.summary[f"{short}/combined_skill"] = combined_skill
            run.summary[f"{short}/cnn_wins"] = metrics["cnn_wins"]
            run.summary[f"{short}/mean_gate"] = metrics["mean_gate"]
            summary_rows.append(
                [
                    label,
                    unit,
                    raw["rmse"],
                    cnn["rmse"],
                    rmse_skill * 100.0,
                    raw["mae"],
                    cnn["mae"],
                    mae_skill * 100.0,
                    combined_skill * 100.0,
                ]
            )

        run.log(
            {
                "aggregate/table": wandb.Table(
                    columns=[
                        "variable",
                        "unit",
                        "raw_iwRMSE",
                        "cnn_iwRMSE",
                        "iwRMSE_skill_percent",
                        "raw_iwMAE",
                        "cnn_iwMAE",
                        "iwMAE_skill_percent",
                        "combined_skill_percent",
                    ],
                    data=summary_rows,
                )
            }
        )

        lead_columns = [
            "lead_hour",
            "variable",
            "raw_iwRMSE",
            "cnn_iwRMSE",
            "raw_iwMAE",
            "cnn_iwMAE",
            "raw_combined",
            "cnn_combined",
        ]
        lead_rows = []
        for row in diagnostics:
            payload = {"lead_hour": row["lead_hour"]}
            for variable, (short, _label, _unit) in VARIABLES.items():
                raw = row["variables"][variable]["raw_gfs"]
                cnn = row["variables"][variable]["cnn_corrected"]
                payload[f"{short}/raw_gfs/iwRMSE"] = raw["rmse"]
                payload[f"{short}/cnn/iwRMSE"] = cnn["rmse"]
                payload[f"{short}/raw_gfs/iwMAE"] = raw["mae"]
                payload[f"{short}/cnn/iwMAE"] = cnn["mae"]
                payload[f"{short}/raw_gfs/combined_error"] = raw[
                    "combined_error"
                ]
                payload[f"{short}/cnn/combined_error"] = cnn["combined_error"]
                lead_rows.append(
                    [
                        row["lead_hour"],
                        variable,
                        raw["rmse"],
                        cnn["rmse"],
                        raw["mae"],
                        cnn["mae"],
                        raw["combined_error"],
                        cnn["combined_error"],
                    ]
                )
            run.log(payload)

        run.log(
            {
                "per_lead/table": wandb.Table(
                    columns=lead_columns,
                    data=lead_rows,
                )
            }
        )

        leads = [row["lead_hour"] for row in diagnostics]
        custom_charts = {}
        for variable, (short, label, unit) in VARIABLES.items():
            for metric, metric_label in (("rmse", "iwRMSE"), ("mae", "iwMAE")):
                raw_values = [
                    row["variables"][variable]["raw_gfs"][metric]
                    for row in diagnostics
                ]
                cnn_values = [
                    row["variables"][variable]["cnn_corrected"][metric]
                    for row in diagnostics
                ]
                custom_charts[f"charts/{short}_{metric_label}"] = (
                    wandb.plot.line_series(
                        xs=[leads, leads],
                        ys=[raw_values, cnn_values],
                        keys=["Raw GFS", "Lead-aware CNN"],
                        title=f"{label} {metric_label} ({unit})",
                        xname="Lead hour",
                    )
                )
        run.log(custom_charts)

        artifact = wandb.Artifact(
            name=f"{run_name}-results",
            type="cnn-diagnostic-results",
            metadata={
                "cycle": cycle["cycle"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            },
        )
        artifact.add_file(str(result_path), name="diagnostic.json")
        if image_path.is_file():
            artifact.add_file(str(image_path), name=image_path.name)
        run.log_artifact(artifact)
        print(json.dumps({"wandb_run": run.url or run.id}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
