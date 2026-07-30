#!/usr/bin/env python3
"""Fit frozen calibrated-GFS coefficients from the locked calibration period.

Learns only from calibration issues in benchmark_v1_selection.json. Never
reads test-cycle forecasts/truth for coefficient estimation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        default="data/evaluation/plans/benchmark_v1_selection.json",
    )
    parser.add_argument(
        "--store-dir",
        default="data/evaluation/forecast_store_hist",
        help="Evaluation-only ForecastStore containing calib native GFS.",
    )
    parser.add_argument(
        "--era5-dir",
        default="data/evaluation/era5",
        help="Root with {variable}/era5_YYYY-MM-DD.nc files.",
    )
    parser.add_argument(
        "--coefficients-out",
        default=(
            "data/evaluation/plans/benchmark_v1_calibrated_gfs_coefficients.json"
        ),
    )
    parser.add_argument("--expected-hotkey")
    parser.add_argument(
        "--commitment-map",
        help=(
            "JSON mapping cycle -> {variable@0_360: {commitment_hash, "
            "manifest_sha256}} or cycle-level hashes."
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help=(
            "Record missing calib cycles as skipped instead of aborting. "
            "Still forbids using any test cycle."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from datetime import timedelta

    from evaluation.artifacts import ForecastArtifactReader
    from evaluation.calibration import (
        BiasAccumulator,
        assert_no_test_cycle_in_fit,
    )
    from evaluation.selection import (
        calibration_issue_times,
        load_selection,
        test_cycle_times,
    )
    from evaluation.truth import Era5TruthLoader

    selection = load_selection(args.selection)
    selection_sha256 = selection["content_sha256"]
    test_keys = [
        cycle.strftime("%Y%m%dT%H%M%SZ")
        for cycle in test_cycle_times(selection)
    ]
    issues = calibration_issue_times(selection)
    issue_keys = [cycle.strftime("%Y%m%dT%H%M%SZ") for cycle in issues]
    assert_no_test_cycle_in_fit(issue_keys, test_keys)

    commitment_map = {}
    if args.commitment_map:
        commitment_map = json.loads(
            Path(args.commitment_map).read_text(encoding="utf-8")
        )

    reader = ForecastArtifactReader(
        args.store_dir,
        require_complete_bundle=False,
    )
    truth_loader = Era5TruthLoader()
    accumulator = BiasAccumulator()
    fitted: dict[str, list[str]] = defaultdict(list)
    skipped: list[dict[str, str]] = []

    variables = tuple(selection["variables"])
    for cycle in issues:
        cycle_key = cycle.strftime("%Y%m%dT%H%M%SZ")
        for variable in variables:
            try:
                hashes = _resolve_hashes(
                    commitment_map,
                    cycle_key,
                    variable,
                    360,
                )
                artifact = reader.read(
                    cycle,
                    variable,
                    360,
                    expected_hotkey=args.expected_hotkey,
                    expected_commitment_hash=hashes["commitment_hash"],
                    expected_manifest_sha256=hashes["manifest_sha256"],
                )
                truth_files = _truth_files_for_window(
                    Path(args.era5_dir),
                    variable,
                    cycle,
                    360,
                )
                truth = truth_loader.load(
                    truth_files,
                    variable=variable,
                    cycle_time=cycle,
                    horizon_hours=360,
                )
                accumulator.update(
                    variable=variable,
                    cycle_time=cycle,
                    forecast=artifact.tensor,
                    truth=truth.tensor,
                )
                fitted[variable].append(cycle_key)
            except Exception as exc:  # noqa: BLE001 - record and continue/abort
                skipped.append(
                    {
                        "cycle": cycle_key,
                        "variable": variable,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                if not args.skip_missing:
                    raise

    assert_no_test_cycle_in_fit(fitted, test_keys)
    frozen = accumulator.freeze(
        selection_sha256=selection_sha256,
        plan_id=selection["plan_id"],
        fitted_issue_cycles=dict(fitted),
    )
    destination = frozen.write(args.coefficients_out)
    print(
        json.dumps(
            {
                "coefficients_path": str(destination.resolve()),
                "coefficients_sha256": frozen.coefficients_sha256,
                "selection_sha256": selection_sha256,
                "fitted_counts": {
                    variable: len(cycles)
                    for variable, cycles in fitted.items()
                },
                "skipped": skipped,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resolve_hashes(
    commitment_map: dict,
    cycle_key: str,
    variable: str,
    horizon: int,
) -> dict[str, str]:
    if not commitment_map:
        raise ValueError(
            "Calibration fit requires --commitment-map with trusted hashes."
        )
    cycle_entry = commitment_map.get(cycle_key)
    if cycle_entry is None:
        raise KeyError(f"No commitment map entry for {cycle_key}.")
    state_key = f"{variable}@0_{horizon}"
    if state_key in cycle_entry:
        entry = cycle_entry[state_key]
    elif "commitment_hash" in cycle_entry and "manifest_sha256" in cycle_entry:
        entry = cycle_entry
    else:
        raise KeyError(
            f"Commitment map for {cycle_key} lacks {state_key} or "
            "cycle-level hashes."
        )
    commitment_hash = str(entry["commitment_hash"])
    manifest_sha256 = str(entry["manifest_sha256"])
    if len(commitment_hash) != 64 or len(manifest_sha256) != 64:
        raise ValueError(f"Invalid hashes for {cycle_key}/{state_key}.")
    return {
        "commitment_hash": commitment_hash,
        "manifest_sha256": manifest_sha256,
    }


def _truth_files_for_window(
    era5_dir: Path,
    variable: str,
    cycle,
    horizon_hours: int,
):
    from datetime import timedelta

    start = cycle.date()
    end = (cycle + timedelta(hours=horizon_hours)).date()
    files = []
    day = start
    while day <= end:
        path = era5_dir / variable / f"era5_{day.isoformat()}.nc"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ERA5 file: {path}")
        files.append(path)
        day = day + timedelta(days=1)
    return files


if __name__ == "__main__":
    raise SystemExit(main())
