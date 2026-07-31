#!/usr/bin/env python3
"""Build the locked 30-cycle × 4-var × 2-horizon backtest plan from hist store."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        default="data/evaluation/plans/benchmark_v1_selection.json",
    )
    parser.add_argument(
        "--store-dir",
        default="data/evaluation/forecast_store_hist",
    )
    parser.add_argument(
        "--era5-dir",
        default="data/evaluation/era5",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation/plans/benchmark_v1_backtest_plan.json",
    )
    parser.add_argument(
        "--require-production-offset",
        type=int,
        default=6,
    )
    args = parser.parse_args()

    from evaluation.selection import canonical_json_bytes, load_selection

    selection = load_selection(args.selection)
    store = Path(args.store_dir)
    era5 = Path(args.era5_dir)
    cases = []
    for item in selection["test_cycles"]:
        cycle_key = item["cycle"]
        manifest_path = store / "bundles" / cycle_key / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"Missing bundle manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        offset = manifest.get("gfs_source_offset_hours")
        if offset != args.require_production_offset:
            raise SystemExit(
                f"{cycle_key} offset={offset}; required "
                f"{args.require_production_offset}"
            )
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        cycle = _parse(cycle_key)
        for variable in selection["variables"]:
            for horizon in selection["horizons_hours"]:
                state_key = f"{variable}@0_{horizon}"
                art = manifest["artifacts"][state_key]
                truth_files = _truth_files(era5, variable, cycle, horizon)
                cases.append(
                    {
                        "cycle": cycle_key,
                        "variable": variable,
                        "horizon": horizon,
                        "commitment_hash": art["commitment_hash"],
                        "manifest_sha256": manifest_sha,
                        "truth_files": [str(path) for path in truth_files],
                    }
                )

    payload = {
        "plan_id": "benchmark_v1",
        "selection_sha256": selection["content_sha256"],
        "store_dir": str(store),
        "n_cases": len(cases),
        "n_cycles": len(selection["test_cycles"]),
        "require_production_offset": args.require_production_offset,
        "cases": cases,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "output": str(out.resolve()),
                "n_cases": len(cases),
                "n_cycles": len(selection["test_cycles"]),
            },
            indent=2,
        )
    )
    return 0


def _parse(cycle_key: str):
    from datetime import datetime, timezone

    return datetime.strptime(cycle_key, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )


def _truth_files(era5: Path, variable: str, cycle, horizon: int) -> list[Path]:
    files = []
    day = cycle.date()
    end = (cycle + timedelta(hours=horizon)).date()
    while day <= end:
        path = era5 / variable / f"era5_{day.isoformat()}.nc"
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
        day = day + timedelta(days=1)
    return files


if __name__ == "__main__":
    raise SystemExit(main())
