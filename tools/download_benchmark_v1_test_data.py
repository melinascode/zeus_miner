#!/usr/bin/env python3
"""Download only the locked benchmark_v1 test-cycle GFS+ERA5 inputs.

Writes under data/evaluation/ only. Records per-cycle download status in
data/evaluation/plans/benchmark_v1_download_status.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_HOTKEY = "5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        default="data/evaluation/plans/benchmark_v1_selection.json",
    )
    parser.add_argument(
        "--status-file",
        default="data/evaluation/plans/benchmark_v1_download_status.json",
    )
    parser.add_argument("--hotkey", default=DEFAULT_HOTKEY)
    parser.add_argument(
        "--store-dir",
        default="data/evaluation/forecast_store_hist",
    )
    parser.add_argument(
        "--era5-dir",
        default="data/evaluation/era5",
    )
    parser.add_argument(
        "--skip-era5",
        action="store_true",
    )
    parser.add_argument(
        "--skip-gfs",
        action="store_true",
    )
    parser.add_argument(
        "--gfs-only-cycle",
        help="If set, build only this one GFS cycle (smoke test).",
    )
    parser.add_argument(
        "--era5-smoke-days",
        type=int,
        default=0,
        help="If >0, download only the first N unique ERA5 days.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from evaluation.selection import load_selection, test_cycle_times

    selection = load_selection(args.selection)
    cycles = test_cycle_times(selection)
    status_path = Path(args.status_file)
    status = _load_status(status_path, selection["content_sha256"], cycles)

    if not args.skip_era5:
        days = sorted(
            {
                (cycle + timedelta(hours=hour)).date()
                for cycle in cycles
                for hour in range(361)
            }
        )
        if args.era5_smoke_days > 0:
            days = days[: args.era5_smoke_days]
        status["era5"]["planned_days"] = len(days)
        status["era5"]["started_at_utc"] = _now()
        _write_status(status_path, status)
        # Download contiguous span covering requested days.
        start = days[0].isoformat()
        end = days[-1].isoformat()
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "fetch_era5_evaluation.py"),
            "--start-date",
            start,
            "--end-date",
            end,
            "--output-dir",
            args.era5_dir,
            "--env-file",
            str(PROJECT_ROOT / "validator.env"),
        ]
        print("START ERA5", start, "→", end, flush=True)
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
        status["era5"]["finished_at_utc"] = _now()
        status["era5"]["exit_code"] = completed.returncode
        status["era5"]["downloaded_files"] = _count_era5_files(
            Path(args.era5_dir), days
        )
        _write_status(status_path, status)
        if completed.returncode != 0 and args.era5_smoke_days == 0:
            print("ERA5 batch reported failures; continuing GFS if requested.")

    if not args.skip_gfs:
        gfs_cycles = cycles
        if args.gfs_only_cycle:
            only = args.gfs_only_cycle
            gfs_cycles = tuple(
                c for c in cycles if c.strftime("%Y%m%dT%H%M%SZ") == only
            )
            if not gfs_cycles:
                raise SystemExit(f"Cycle {only} is not in locked selection.")
        for cycle in gfs_cycles:
            cycle_key = cycle.strftime("%Y%m%dT%H%M%SZ")
            entry = status["gfs"].setdefault(
                cycle_key,
                {"status": "pending", "failure_reason": None},
            )
            if entry.get("status") == "complete":
                print(f"SKIP GFS {cycle_key} already complete", flush=True)
                continue
            print(f"START GFS {cycle_key}", flush=True)
            entry["status"] = "running"
            entry["started_at_utc"] = _now()
            _write_status(status_path, status)
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "build_historical_gfs_bundle.py"),
                "--target-cycle",
                cycle_key,
                "--gfs-cycle",
                cycle_key,
                "--hotkey",
                args.hotkey,
                "--store-dir",
                args.store_dir,
                "--cache-dir",
                "data/evaluation/gfs_cache",
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
            entry["elapsed_seconds"] = round(time.time() - t0, 1)
            entry["finished_at_utc"] = _now()
            if completed.returncode == 0:
                entry["status"] = "complete"
                entry["failure_reason"] = None
            else:
                entry["status"] = "failed"
                entry["failure_reason"] = f"exit_code={completed.returncode}"
            _write_status(status_path, status)
            print(
                f"DONE GFS {cycle_key} status={entry['status']} "
                f"elapsed_s={entry['elapsed_seconds']}",
                flush=True,
            )

    print(json.dumps({"status_file": str(status_path.resolve())}, indent=2))
    return 0


def _load_status(path: Path, selection_sha256: str, cycles) -> dict:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("selection_sha256") != selection_sha256:
            raise SystemExit(
                "Download status file is bound to a different selection hash."
            )
        return payload
    return {
        "schema_version": 1,
        "plan_id": "benchmark_v1",
        "selection_sha256": selection_sha256,
        "mode": "test_only",
        "era5": {},
        "gfs": {
            c.strftime("%Y%m%dT%H%M%SZ"): {
                "status": "pending",
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


def _count_era5_files(era5_dir: Path, days) -> int:
    count = 0
    for variable in (
        "2m_temperature",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "surface_solar_radiation_downwards",
    ):
        for day in days:
            if (era5_dir / variable / f"era5_{day.isoformat()}.nc").is_file():
                count += 1
    return count


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
