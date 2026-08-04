from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

from forecast.forecast_store import ForecastStore
from forecast.gfs_bundle_provider import GFSBundleLeadProvider
from tools.precompute_gfs_bundle import parse_cycle, utc_text
from tools.precompute_gfs_native_store import (
    EXPECTED_STATE_KEYS,
    run as run_native_precompute,
)
from tools.select_gfs_source_cycle import (
    parse_offsets,
    select_source_cycle,
)


def parse_iso_utc(value: str) -> datetime:
    return datetime.strptime(
        value,
        "%Y-%m-%dT%H:%M:%SZ",
    )


def atomic_write_json(
    destination: Path,
    document: dict[str, Any],
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )

    temporary.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        destination,
    )


def load_existing(
    store: ForecastStore,
    target_cycle: datetime,
    hotkey: str,
) -> dict[str, Any] | None:
    try:
        return store.load_manifest(
            target_cycle,
            expected_state_keys=EXPECTED_STATE_KEYS,
            verify_files=True,
            expected_hotkey=hotkey,
        )
    except FileNotFoundError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the newest complete GFS source "
            "cycle and publish a schema-v2 Zeus "
            "ForecastStore bundle."
        )
    )

    parser.add_argument(
        "--target-cycle",
        required=True,
        type=parse_cycle,
        help=(
            "Zeus target cycle in "
            "YYYYMMDDTHHMMSSZ format."
        ),
    )

    parser.add_argument(
        "--hotkey",
        required=True,
    )

    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--store-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--work-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--selection-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--offsets",
        default="6,12,18,24",
    )

    parser.add_argument(
        "--retention-days",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--max-run-age-hours",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Select and validate a source cycle "
            "without generating artifacts."
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    store = ForecastStore(
        directory=args.store_dir,
        retention_days=args.retention_days,
    )

    target_key = store.cycle_key(
        args.target_cycle
    )

    print("=" * 100)
    print("ZEUS GFS PRECOMPUTE ORCHESTRATOR")
    print("=" * 100)
    print(
        "Target cycle:",
        utc_text(args.target_cycle),
    )
    print(
        "Native store:",
        args.store_dir.resolve(),
    )
    print(
        "Dry run:",
        args.dry_run,
    )

    existing = load_existing(
        store=store,
        target_cycle=args.target_cycle,
        hotkey=args.hotkey,
    )

    if existing is not None:
        print()
        print(
            "SKIP: complete native ForecastStore "
            f"bundle already exists for {target_key}"
        )
        print(
            "Artifacts:",
            len(existing["artifacts"]),
        )
        print(
            "Source GFS cycle:",
            existing.get(
                "gfs_common_cycle_utc"
            ),
        )
        print()
        print(
            "PASS: completed target cycle "
            "requires no precompute work"
        )
        return

    offsets = parse_offsets(
        args.offsets
    )

    provider = GFSBundleLeadProvider(
        cache_directory=args.cache_dir,
        max_run_age_hours=(
            args.max_run_age_hours
        ),
    )

    selection, attempts = select_source_cycle(
        provider=provider,
        target_cycle=args.target_cycle,
        offsets=offsets,
    )

    selection_document = {
        "target_cycle_utc": utc_text(
            args.target_cycle
        ),
        "selection": selection,
        "attempts": attempts,
    }

    atomic_write_json(
        args.selection_json,
        selection_document,
    )

    print()
    print("Selected source:")
    print(
        "  cycle:",
        selection["source_cycle_utc"],
    )
    print(
        "  offset:",
        selection["source_offset_hours"],
        "hours",
    )
    print(
        "  range:",
        f'F{selection["source_forecast_hour_start"]:03d}-'
        f'F{selection["source_forecast_hour_end"]:03d}',
    )
    print(
        "  selection JSON:",
        args.selection_json.resolve(),
    )

    if args.dry_run:
        print()
        print(
            "PASS: source cycle selected; "
            "dry run performed no publication"
        )
        return

    source_cycle = parse_iso_utc(
        selection["source_cycle_utc"]
    )

    native_args = Namespace(
        target_cycle=args.target_cycle,
        gfs_cycle=source_cycle,
        hotkey=args.hotkey,
        cache_dir=args.cache_dir,
        store_dir=args.store_dir,
        work_dir=args.work_dir,
        retention_days=args.retention_days,
        max_run_age_hours=(
            args.max_run_age_hours
        ),
    )

    manifest = run_native_precompute(
        native_args
    )

    verified = store.load_manifest(
        args.target_cycle,
        expected_state_keys=EXPECTED_STATE_KEYS,
        verify_files=True,
        expected_hotkey=args.hotkey,
    )

    if len(verified["artifacts"]) != 8:
        raise RuntimeError(
            "Published target bundle does not "
            "contain eight artifacts."
        )

    if (
        verified["gfs_common_cycle_utc"]
        != selection["source_cycle_utc"]
    ):
        raise RuntimeError(
            "Published manifest source cycle does "
            "not match the selected GFS cycle."
        )

    if (
        verified["gfs_source_offset_hours"]
        != selection["source_offset_hours"]
    ):
        raise RuntimeError(
            "Published source offset does not match "
            "the selector result."
        )

    print()
    print("=" * 100)
    print("ORCHESTRATOR RESULT")
    print("=" * 100)
    print(
        "Target cycle:",
        verified["cycle_key"],
    )
    print(
        "Source cycle:",
        verified[
            "gfs_common_cycle_utc"
        ],
    )
    print(
        "Source offset:",
        verified[
            "gfs_source_offset_hours"
        ],
    )
    print(
        "Artifacts:",
        len(verified["artifacts"]),
    )
    print(
        "Total seconds:",
        verified[
            "precompute_total_seconds"
        ],
    )
    print()
    print(
        "PASS: source selection and native "
        "bundle publication completed"
    )


if __name__ == "__main__":
    main()
