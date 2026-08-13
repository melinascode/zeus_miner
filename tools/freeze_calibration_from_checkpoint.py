#!/usr/bin/env python3
"""Force-freeze calibrated-GFS coefficients from an accumulator checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
        "--status-file",
        default="data/evaluation/plans/benchmark_v1_calib_stream_status.json",
    )
    parser.add_argument(
        "--coefficients-out",
        default=(
            "data/evaluation/plans/benchmark_v1_calibrated_gfs_coefficients.json"
        ),
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Overwrite an existing frozen coefficients file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from evaluation.calibration import BiasAccumulator
    from evaluation.selection import load_selection, test_cycle_times
    from tools.run_streaming_calibration import _load_checkpoint, _write_status

    selection = load_selection(args.selection)
    selection_sha256 = selection["content_sha256"]
    test_keys = [
        c.strftime("%Y%m%dT%H%M%SZ") for c in test_cycle_times(selection)
    ]

    checkpoint_path = Path(args.coefficients_out).with_suffix(".accumulator.npz")
    meta_path = Path(str(checkpoint_path) + ".meta.json")
    if not checkpoint_path.is_file() or not meta_path.is_file():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    accumulator = BiasAccumulator()
    fitted, accumulator = _load_checkpoint(checkpoint_path, meta_path, accumulator)
    counts = {variable: len(cycles) for variable, cycles in fitted.items()}
    if min(counts.values()) < 1:
        raise SystemExit(f"Checkpoint has no fitted samples: {counts}")

    frozen = accumulator.freeze(
        selection_sha256=selection_sha256,
        plan_id=selection["plan_id"],
        fitted_issue_cycles=dict(fitted),
    )
    frozen.write(args.coefficients_out, allow_overwrite=args.allow_overwrite)

    status_path = Path(args.status_file)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    complete = sum(
        1 for entry in status.get("cycles", {}).values() if entry.get("status") == "complete"
    )
    failed = sum(
        1 for entry in status.get("cycles", {}).values() if entry.get("status") == "failed"
    )
    status["coefficients_path"] = args.coefficients_out
    status["coefficients_sha256"] = frozen.coefficients_sha256
    status["finished_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    status["fitted_counts"] = counts
    status["summary"] = {
        "frozen": True,
        "coefficients_sha256": frozen.coefficients_sha256,
        "fitted_counts": counts,
        "complete_cycles": complete,
        "failed_cycles": failed,
        "forced_from_checkpoint": True,
    }
    _write_status(status_path, status)
    checkpoint_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    print(json.dumps(status["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
