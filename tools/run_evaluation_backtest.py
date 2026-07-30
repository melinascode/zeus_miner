#!/usr/bin/env python3
"""Run a deterministic multi-cycle validator-faithful backtest plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        help="JSON file containing a cases list.",
    )
    parser.add_argument(
        "--selection",
        help=(
            "Locked benchmark_v1_selection.json. When set, verifies the "
            "immutable hash before scoring."
        ),
    )
    parser.add_argument(
        "--registry",
        default="data/evaluation/plans/benchmark_v1_registry.json",
        help="Mutable cycle status registry bound to the selection hash.",
    )
    parser.add_argument(
        "--coefficients",
        help=(
            "Frozen calibrated-GFS coefficients JSON. Enables the third model."
        ),
    )
    parser.add_argument(
        "--store-dir",
        default="data/forecast_store_v2",
    )
    parser.add_argument(
        "--output-dir",
        default="data/evaluation",
    )
    parser.add_argument("--expected-hotkey")
    parser.add_argument("--minimum-cycles", type=int, default=1)
    parser.add_argument("--require-full-matrix", action="store_true")
    parser.add_argument("--require-independent", action="store_true")
    parser.add_argument("--lead-diagnostics", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Record failed/incomplete cases in the registry instead of "
            "aborting the whole plan."
        ),
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.plan:
        raise SystemExit("--plan is required.")

    from evaluation.artifacts import ForecastArtifactReader
    from evaluation.backtest import (
        load_backtest_plan,
        summarize_results,
        validate_backtest_plan,
    )
    from evaluation.calibration import FrozenCalibration
    from evaluation.runner import evaluate_case, write_evaluation_result
    from evaluation.scoring import ValidatorFaithfulScorer
    from evaluation.selection import (
        assert_selection_unchanged,
        load_selection,
        record_cycle_status,
    )
    from evaluation.truth import Era5TruthLoader
    from evaluation.wandb_launcher import launch_wandb_logger

    selection = None
    selection_sha256 = None
    if args.selection:
        selection = load_selection(args.selection)
        selection_sha256 = selection["content_sha256"]

    calibration = None
    if args.coefficients:
        if selection_sha256 is None:
            raise SystemExit(
                "--coefficients requires --selection so leakage checks bind "
                "to the locked plan."
            )
        calibration = FrozenCalibration.load(
            args.coefficients,
            expected_selection_sha256=selection_sha256,
        )

    cases = validate_backtest_plan(
        load_backtest_plan(args.plan),
        minimum_cycles=args.minimum_cycles,
        require_full_matrix=args.require_full_matrix,
        require_independent=args.require_independent,
    )
    if selection is not None:
        allowed = {
            item["cycle"] for item in selection["test_cycles"]
        }
        for case in cases:
            cycle_key = case.cycle_time.strftime("%Y%m%dT%H%M%SZ")
            if cycle_key not in allowed:
                raise ValueError(
                    f"Plan case {cycle_key} is not in the locked selection."
                )
        assert_selection_unchanged(args.selection, selection_sha256)

    reader = ForecastArtifactReader(args.store_dir)
    truth_loader = Era5TruthLoader()
    scorer = ValidatorFaithfulScorer()

    results = []
    result_paths = []
    incomplete = []
    wandb_runs = []
    seen_payloads: dict[tuple[str, int, str], str] = {}
    cycle_outcomes: dict[str, list[str]] = {}
    for case in cases:
        cycle_key = case.cycle_time.strftime("%Y%m%dT%H%M%SZ")
        try:
            if selection_sha256 is not None:
                assert_selection_unchanged(
                    args.selection,
                    selection_sha256,
                )
            artifact = reader.read(
                case.cycle_time,
                case.variable,
                case.horizon_hours,
                expected_hotkey=args.expected_hotkey,
                expected_commitment_hash=case.commitment_hash,
                expected_manifest_sha256=case.manifest_sha256,
            )
            if args.require_independent:
                payload_key = (
                    case.variable,
                    case.horizon_hours,
                    artifact.payload_sha256,
                )
                previous_cycle = seen_payloads.get(payload_key)
                if previous_cycle is not None:
                    raise ValueError(
                        "Independent backtest contains duplicate forecast "
                        f"bytes for {case.variable}@0_{case.horizon_hours}: "
                        f"{previous_cycle} and "
                        f"{case.cycle_time.isoformat()}."
                    )
                seen_payloads[payload_key] = case.cycle_time.isoformat()
            truth = truth_loader.load(
                case.truth_files,
                variable=case.variable,
                cycle_time=case.cycle_time,
                horizon_hours=case.horizon_hours,
            )
            result = evaluate_case(
                artifact,
                truth,
                scorer=scorer,
                include_lead_diagnostics=args.lead_diagnostics,
                calibration=calibration,
                selection_sha256=selection_sha256,
            )
            result_path = write_evaluation_result(result, args.output_dir)
            results.append(result)
            result_paths.append(str(result_path.resolve()))
            cycle_outcomes.setdefault(cycle_key, []).append("complete")
            if args.wandb:
                wandb_runs.append(
                    launch_wandb_logger(result_path, args.wandb_mode)[
                        "wandb_run"
                    ]
                )
        except Exception as exc:  # noqa: BLE001 - retain failed cycles
            record = {
                "cycle": cycle_key,
                "variable": case.variable,
                "horizon_hours": case.horizon_hours,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            incomplete.append(record)
            cycle_outcomes.setdefault(cycle_key, []).append("failed")
            if not args.continue_on_error:
                if args.selection:
                    record_cycle_status(
                        args.registry,
                        cycle_key,
                        "failed",
                        failure_reason=record["reason"],
                        selection_sha256=selection_sha256,
                    )
                raise

    if args.selection:
        for cycle_key, outcomes in cycle_outcomes.items():
            if all(item == "complete" for item in outcomes):
                status = "complete"
                reason = None
            elif any(item == "complete" for item in outcomes):
                status = "incomplete"
                reason = "partial matrix failure"
            else:
                status = "failed"
                reason = next(
                    (
                        item["reason"]
                        for item in incomplete
                        if item["cycle"] == cycle_key
                    ),
                    "cycle failed",
                )
            record_cycle_status(
                args.registry,
                cycle_key,
                status,
                failure_reason=reason,
                selection_sha256=selection_sha256,
            )

    if selection_sha256 is not None:
        assert_selection_unchanged(args.selection, selection_sha256)

    summary = summarize_results(results, incomplete=incomplete)
    summary["plan"] = {
        "path": str(Path(args.plan).resolve()),
        "minimum_cycles": args.minimum_cycles,
        "require_full_matrix": args.require_full_matrix,
        "require_independent": args.require_independent,
        "selection": (
            str(Path(args.selection).resolve()) if args.selection else None
        ),
        "selection_sha256": selection_sha256,
        "coefficients": (
            str(Path(args.coefficients).resolve())
            if args.coefficients
            else None
        ),
        "coefficients_sha256": (
            calibration.coefficients_sha256 if calibration else None
        ),
    }
    summary["result_paths"] = result_paths
    summary["wandb_runs"] = wandb_runs
    summary_path = _write_summary(summary, Path(args.output_dir))
    print(
        json.dumps(
            {
                "summary_path": str(summary_path.resolve()),
                "unique_cycles": summary["unique_cycles"],
                "evaluations": summary["evaluations"],
                "incomplete_or_failed_count": summary[
                    "incomplete_or_failed_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _write_summary(summary: dict, output_directory: Path) -> Path:
    content = json.dumps(
        summary,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()[:16]
    destination = output_directory / f"backtest-summary-{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    raise SystemExit(main())
