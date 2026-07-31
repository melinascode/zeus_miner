#!/usr/bin/env python3
"""Rebuild locked test-cycle hist GFS bundles with production source selection.

For each target cycle:
  1. Enforce ≥80 GiB free (plus a small reserve).
  2. Select newest ready source among offsets (6,12,18,24) — same as
     tools/select_gfs_source_cycle.py / production.
  3. Remove any existing non-faithful (or any) bundle for that target.
  4. Build via tools/build_historical_gfs_bundle.py with --gfs-cycle set.
  5. Verify manifest offset matches the selection.

Writes status to data/evaluation/plans/benchmark_v1_gfs_fidelity_rebuild.json.
Never touches data/forecast_store_v2 or the live miner.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_HOTKEY = "5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR"
DEFAULT_STORE = Path("data/evaluation/forecast_store_hist")
DEFAULT_CACHE = Path("data/evaluation/gfs_cache")
DEFAULT_STATUS = Path(
    "data/evaluation/plans/benchmark_v1_gfs_fidelity_rebuild.json"
)
PRODUCTION_OFFSETS = (6, 12, 18, 24)
MIN_FREE_GIB = 80.0
RESERVE_GIB = 6.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        default="data/evaluation/plans/benchmark_v1_selection.json",
    )
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--hotkey", default=DEFAULT_HOTKEY)
    parser.add_argument(
        "--only-cycle",
        help="Rebuild a single cycle key (YYYYMMDDTHHMMSSZ).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Rebuild at most N cycles (0 = all pending).",
    )
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        default=True,
        help="Skip cycles already marked production_faithful complete (default).",
    )
    parser.add_argument(
        "--no-skip-complete",
        action="store_false",
        dest="skip_complete",
    )
    parser.add_argument(
        "--assume-offset",
        type=int,
        choices=PRODUCTION_OFFSETS,
        help=(
            "Skip readiness probing and use this offset (historical archives "
            "are normally always ready at 6). Still verifies after build."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from evaluation.disk_safety import assert_disk_safety, free_gib
    from evaluation.selection import load_selection, test_cycle_times
    from tools.precompute_gfs_bundle import parse_cycle, utc_text
    from tools.select_gfs_source_cycle import select_source_cycle

    selection = load_selection(args.selection)
    cycles = test_cycle_times(selection)
    if args.only_cycle:
        cycles = tuple(
            c
            for c in cycles
            if c.strftime("%Y%m%dT%H%M%SZ") == args.only_cycle
        )
        if not cycles:
            raise SystemExit(f"{args.only_cycle} is not in the locked selection.")

    status = _load_status(
        args.status_file,
        selection["content_sha256"],
        cycles if not args.only_cycle else test_cycle_times(selection),
    )
    status["selection_sha256"] = selection["content_sha256"]
    status["mode"] = "production_source_rebuild"
    status["started_at_utc"] = status.get("started_at_utc") or _now()

    rebuilt = 0
    for target in cycles:
        if args.limit and rebuilt >= args.limit:
            break
        cycle_key = target.strftime("%Y%m%dT%H%M%SZ")
        entry = status["cycles"].setdefault(
            cycle_key,
            {
                "status": "pending",
                "production_faithful": False,
                "failure_reason": None,
            },
        )
        if (
            args.skip_complete
            and entry.get("status") == "complete"
            and entry.get("production_faithful") is True
        ):
            print(f"SKIP {cycle_key} already production-faithful", flush=True)
            continue

        print("=" * 80, flush=True)
        print(f"REBUILD {cycle_key}", flush=True)
        _maybe_prune_cache(args.cache_dir, keep_dates=_nearby_dates(target))
        free_before = assert_disk_safety(
            PROJECT_ROOT,
            min_free_gib=MIN_FREE_GIB,
            reserve_for_operation_gib=RESERVE_GIB,
        )
        entry["status"] = "running"
        entry["started_at_utc"] = _now()
        entry["free_gib_before"] = round(free_before, 2)
        _write_status(args.status_file, status)

        try:
            if args.assume_offset is not None:
                offset = int(args.assume_offset)
                source = target - timedelta(hours=offset)
                selection_doc = {
                    "target_cycle_utc": utc_text(target),
                    "source_cycle_utc": utc_text(source),
                    "source_offset_hours": offset,
                    "assumed": True,
                }
                attempts: list = []
            else:
                from forecast.gfs_bundle_provider import GFSBundleLeadProvider

                provider = GFSBundleLeadProvider(
                    cache_directory=args.cache_dir,
                    max_run_age_hours=24 * 500,
                    sflux_priority=("aws", "nomads"),
                )
                selection_doc, attempts = select_source_cycle(
                    provider=provider,
                    target_cycle=target,
                    offsets=PRODUCTION_OFFSETS,
                )
            source_key = _cycle_key_from_utc_text(
                selection_doc["source_cycle_utc"]
            )
            offset = int(selection_doc["source_offset_hours"])
            if offset not in PRODUCTION_OFFSETS:
                raise RuntimeError(
                    f"Selected offset {offset} is not a production offset."
                )
            entry["selection"] = selection_doc
            entry["selection_attempts"] = attempts
            _write_status(args.status_file, status)

            _remove_bundle(args.store_dir, cycle_key)

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "build_historical_gfs_bundle.py"),
                "--target-cycle",
                cycle_key,
                "--gfs-cycle",
                source_key,
                "--hotkey",
                args.hotkey,
                "--store-dir",
                str(args.store_dir),
                "--cache-dir",
                str(args.cache_dir),
                "--work-dir",
                f"data/evaluation/gfs_work/{cycle_key}",
                "--retention-days",
                "500",
                "--max-run-age-hours",
                str(24 * 500),
                "--sflux-priority",
                "aws,nomads",
            ]
            t0 = time.time()
            completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
            elapsed = round(time.time() - t0, 1)
            entry["elapsed_seconds"] = elapsed
            entry["finished_at_utc"] = _now()
            if completed.returncode != 0:
                entry["status"] = "failed"
                entry["failure_reason"] = f"exit_code={completed.returncode}"
                entry["production_faithful"] = False
                _write_status(args.status_file, status)
                print(f"FAIL {cycle_key} exit={completed.returncode}", flush=True)
                continue

            manifest_path = (
                args.store_dir / "bundles" / cycle_key / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            got_offset = manifest.get("gfs_source_offset_hours")
            got_source = manifest.get("gfs_common_cycle_utc")
            faithful = (
                got_offset == offset
                and _normalize_cycle_utc(got_source)
                == _normalize_cycle_utc(selection_doc["source_cycle_utc"])
            )
            entry["manifest_gfs_source_offset_hours"] = got_offset
            entry["manifest_gfs_common_cycle_utc"] = got_source
            entry["production_faithful"] = bool(faithful)
            if not faithful:
                entry["status"] = "failed"
                entry["failure_reason"] = (
                    f"manifest offset/source mismatch: got offset={got_offset} "
                    f"source={got_source}; expected offset={offset} "
                    f"source={selection_doc['source_cycle_utc']}"
                )
                _write_status(args.status_file, status)
                print(f"FAIL {cycle_key} fidelity mismatch", flush=True)
                continue

            entry["status"] = "complete"
            entry["failure_reason"] = None
            entry["free_gib_after"] = round(free_gib(PROJECT_ROOT), 2)
            _write_status(args.status_file, status)
            rebuilt += 1
            print(
                f"PASS {cycle_key} offset={offset} source={source_key} "
                f"elapsed_s={elapsed} free_gib={entry['free_gib_after']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — per-cycle isolation
            entry["status"] = "failed"
            entry["failure_reason"] = f"{type(exc).__name__}: {exc}"
            entry["production_faithful"] = False
            entry["finished_at_utc"] = _now()
            _write_status(args.status_file, status)
            print(f"FAIL {cycle_key}: {entry['failure_reason']}", flush=True)

    status["finished_at_utc"] = _now()
    status["summary"] = _summarize(status)
    _write_status(args.status_file, status)
    print(json.dumps(status["summary"], indent=2, sort_keys=True), flush=True)
    failed = status["summary"]["failed"]
    return 1 if failed else 0


def _remove_bundle(store_dir: Path, cycle_key: str) -> None:
    bundle = store_dir / "bundles" / cycle_key
    if bundle.exists():
        print(f"Removing existing bundle {bundle}", flush=True)
        shutil.rmtree(bundle)
    latest = store_dir / "latest.json"
    if latest.is_file():
        try:
            pointer = json.loads(latest.read_text(encoding="utf-8"))
            if pointer.get("cycle_key") == cycle_key:
                latest.unlink()
                print("Cleared latest.json pointer for removed bundle", flush=True)
        except (json.JSONDecodeError, OSError):
            pass


def _nearby_dates(target: datetime) -> set[str]:
    # Keep target day and up to 24h earlier (production max offset).
    dates = set()
    for hours in (0, 6, 12, 18, 24):
        dates.add((target - timedelta(hours=hours)).strftime("%Y%m%d"))
    return dates


def _maybe_prune_cache(cache_dir: Path, keep_dates: set[str]) -> None:
    from evaluation.disk_safety import free_gib

    free = free_gib(PROJECT_ROOT)
    if free >= MIN_FREE_GIB + RESERVE_GIB + 10:
        return
    gfs_root = cache_dir / "gfs"
    if not gfs_root.is_dir():
        return
    removed = 0
    for path in sorted(gfs_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name in keep_dates:
            continue
        print(f"Pruning cache {path} (free_gib={free:.1f})", flush=True)
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
        free = free_gib(PROJECT_ROOT)
        if free >= MIN_FREE_GIB + RESERVE_GIB + 15:
            break
    if removed:
        print(f"Pruned {removed} cache day dirs; free_gib={free:.1f}", flush=True)


def _load_status(path: Path, selection_sha256: str, cycles) -> dict:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("selection_sha256") not in (None, selection_sha256):
            raise SystemExit(
                "Fidelity rebuild status is bound to a different selection hash."
            )
        payload.setdefault("cycles", {})
        for cycle in cycles:
            key = cycle.strftime("%Y%m%dT%H%M%SZ")
            payload["cycles"].setdefault(
                key,
                {
                    "status": "pending",
                    "production_faithful": False,
                    "failure_reason": None,
                },
            )
        return payload
    return {
        "schema_version": 1,
        "plan_id": "benchmark_v1",
        "selection_sha256": selection_sha256,
        "mode": "production_source_rebuild",
        "production_offsets": list(PRODUCTION_OFFSETS),
        "disk_safety_min_free_gib": MIN_FREE_GIB,
        "cycles": {
            c.strftime("%Y%m%dT%H%M%SZ"): {
                "status": "pending",
                "production_faithful": False,
                "failure_reason": None,
            }
            for c in cycles
        },
    }


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _summarize(status: dict) -> dict:
    cycles = status.get("cycles", {})
    counts = {"complete": 0, "failed": 0, "pending": 0, "running": 0}
    faithful = 0
    for entry in cycles.values():
        state = entry.get("status", "pending")
        counts[state] = counts.get(state, 0) + 1
        if entry.get("production_faithful"):
            faithful += 1
    return {
        "n_cycles": len(cycles),
        "production_faithful": faithful,
        **counts,
    }


def _cycle_key_from_utc_text(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    return parsed.strftime("%Y%m%dT%H%M%SZ")


def _normalize_cycle_utc(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
