#!/usr/bin/env python3
"""Build one historical native-GFS Zeus bundle into the evaluation-only store.

Writes exclusively under data/evaluation/. Never touches data/forecast_store_v2
or the production miner pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_STORE = Path("data/evaluation/forecast_store_hist")
DEFAULT_CACHE = Path("data/evaluation/gfs_cache")
DEFAULT_WORK = Path("data/evaluation/gfs_work")
FORBIDDEN_STORE = Path("data/forecast_store_v2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a native GFS ForecastStore v2 bundle for offline "
            "evaluation. Output is restricted to data/evaluation/."
        )
    )
    parser.add_argument("--target-cycle", required=True)
    parser.add_argument(
        "--gfs-cycle",
        help="Defaults to the target cycle (native alignment).",
    )
    parser.add_argument("--hotkey", required=True)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--retention-days", type=int, default=500)
    parser.add_argument("--max-run-age-hours", type=int, default=24 * 400)
    parser.add_argument(
        "--sflux-priority",
        default="aws,nomads",
        help="Historical archives require AWS; NOMADS alone is insufficient.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store_dir = args.store_dir.resolve()
    forbidden = (PROJECT_ROOT / FORBIDDEN_STORE).resolve()
    if store_dir == forbidden or forbidden in store_dir.parents:
        raise SystemExit(
            "Refusing to write historical evaluation bundles into the "
            f"production store {FORBIDDEN_STORE}."
        )
    if "data/evaluation" not in store_dir.as_posix():
        raise SystemExit(
            "Historical evaluation bundles must live under data/evaluation/."
        )

    from tools.precompute_gfs_bundle import parse_cycle
    from tools.precompute_gfs_native_store import run

    target = parse_cycle(args.target_cycle)
    gfs_cycle = parse_cycle(args.gfs_cycle or args.target_cycle)
    namespace = argparse.Namespace(
        target_cycle=target,
        gfs_cycle=gfs_cycle,
        hotkey=args.hotkey,
        cache_dir=args.cache_dir,
        store_dir=args.store_dir,
        work_dir=args.work_dir,
        retention_days=args.retention_days,
        max_run_age_hours=args.max_run_age_hours,
        sflux_priority=[
            part.strip()
            for part in str(args.sflux_priority).split(",")
            if part.strip()
        ],
    )
    run(namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
