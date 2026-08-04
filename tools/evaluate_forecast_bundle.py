#!/usr/bin/env python3
"""Evaluate an immutable Zeus bundle against existing local ERA5 files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_HORIZONS = (48, 360)
SUPPORTED_VARIABLES = (
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one stored Zeus forecast artifact with a persistence "
            "baseline using exact validator RMSE/MAE behavior. This command "
            "never downloads ERA5 or writes into the forecast store."
        )
    )
    parser.add_argument(
        "--store-dir",
        default="data/forecast_store_v2",
        help="Read-only ForecastStore v2 directory.",
    )
    parser.add_argument(
        "--cycle",
        required=True,
        help="Cycle key, ISO UTC timestamp, or latest.",
    )
    parser.add_argument(
        "--variable",
        required=True,
        choices=sorted(SUPPORTED_VARIABLES),
    )
    parser.add_argument(
        "--horizon",
        required=True,
        type=int,
        choices=SUPPORTED_HORIZONS,
        help="Inclusive ending lead hour: 48 or 360.",
    )
    parser.add_argument(
        "--truth-file",
        required=True,
        action="append",
        help=(
            "Existing ERA5 NetCDF file. Repeat for every day needed by the "
            "evaluation window."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/evaluation",
        help="Evaluation output root.",
    )
    parser.add_argument(
        "--expected-hotkey",
        help="Require the forecast bundle to be bound to this hotkey.",
    )
    parser.add_argument(
        "--expected-commitment-hash",
        required=True,
        help="Trusted on-chain commitment hash for the selected artifact.",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="Trusted SHA-256 of the exact manifest.json bytes.",
    )
    parser.add_argument(
        "--lead-diagnostics",
        action="store_true",
        help="Compute additional non-canonical per-lead metrics.",
    )
    parser.add_argument(
        "--selection",
        help="Locked benchmark selection JSON (required with --coefficients).",
    )
    parser.add_argument(
        "--coefficients",
        help="Frozen calibrated-GFS coefficients JSON.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log the paired result from /root/miniconda3/envs/zeus-eval.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from evaluation.artifacts import ForecastArtifactReader
    from evaluation.calibration import FrozenCalibration
    from evaluation.runner import evaluate_case, write_evaluation_result
    from evaluation.selection import load_selection
    from evaluation.truth import Era5TruthLoader
    from evaluation.wandb_launcher import launch_wandb_logger

    selection_sha256 = None
    calibration = None
    if args.coefficients:
        if not args.selection:
            raise SystemExit("--coefficients requires --selection.")
        selection = load_selection(args.selection)
        selection_sha256 = selection["content_sha256"]
        calibration = FrozenCalibration.load(
            args.coefficients,
            expected_selection_sha256=selection_sha256,
        )

    reader = ForecastArtifactReader(args.store_dir)
    artifact = reader.read(
        args.cycle,
        args.variable,
        args.horizon,
        expected_hotkey=args.expected_hotkey,
        expected_commitment_hash=args.expected_commitment_hash,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    truth = Era5TruthLoader().load(
        args.truth_file,
        variable=artifact.variable,
        cycle_time=artifact.cycle_time,
        horizon_hours=artifact.horizon_hours,
    )
    result = evaluate_case(
        artifact,
        truth,
        include_lead_diagnostics=args.lead_diagnostics,
        calibration=calibration,
        selection_sha256=selection_sha256,
    )
    result_path = write_evaluation_result(result, args.output_dir)

    response = {
        "result_path": str(result_path.resolve()),
        "raw_gfs": result["metrics"]["raw_gfs"],
        "persistence": result["metrics"]["persistence"],
        "comparison": result["metrics"]["comparison"],
    }
    if "calibrated_gfs" in result["metrics"]:
        response["calibrated_gfs"] = result["metrics"]["calibrated_gfs"]
    if args.wandb:
        response.update(launch_wandb_logger(result_path, args.wandb_mode))
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
