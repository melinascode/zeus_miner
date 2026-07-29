from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tools.zeus_precompute_schedule import (
    parse_utc,
    resolve_schedule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the scheduled Zeus target cycle, "
            "select a complete GFS source cycle, and "
            "publish its native ForecastStore bundle."
        )
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
        "--metadata-dir",
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
        "--max-lateness-minutes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--now",
        type=parse_utc,
        help=(
            "Test-only UTC override in "
            "YYYYMMDDTHHMMSSZ format."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    metadata_directory = (
        args.metadata_dir.resolve()
    )

    metadata_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_path = (
        metadata_directory
        / ".scheduled-precompute.lock"
    )

    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX
                | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            print(
                "SKIP: another scheduled GFS "
                "precompute process holds the lock"
            )
            return

        now = (
            args.now
            if args.now is not None
            else datetime.now(timezone.utc)
        )

        schedule = resolve_schedule(
            now,
            max_lateness=timedelta(
                minutes=(
                    args.max_lateness_minutes
                )
            ),
        )

        target_key = schedule[
            "target_cycle_key"
        ]

        schedule_path = (
            metadata_directory
            / f"{target_key}.schedule.json"
        )

        selection_path = (
            metadata_directory
            / f"{target_key}.selection.json"
        )

        atomic_write_json(
            schedule_path,
            schedule,
        )

        print("=" * 100)
        print("SCHEDULED ZEUS GFS PRECOMPUTE")
        print("=" * 100)
        print(
            "Invocation time:",
            schedule["now_utc"],
        )
        print(
            "Nominal run time:",
            schedule["nominal_run_utc"],
        )
        print(
            "Scheduled target cycle:",
            target_key,
        )
        print(
            "Commit window opens:",
            schedule[
                "commit_window_open_utc"
            ],
        )
        print(
            "Commit deadline:",
            schedule[
                "commit_deadline_utc"
            ],
        )
        print(
            "Lateness seconds:",
            schedule["lateness_seconds"],
        )
        print(
            "Schedule JSON:",
            schedule_path,
        )
        print(
            "Selection JSON:",
            selection_path,
        )
        print(
            "Dry run:",
            args.dry_run,
        )

        command = [
            sys.executable,
            str(
                PROJECT_ROOT
                / "tools"
                / "run_gfs_precompute_cycle.py"
            ),
            "--target-cycle",
            target_key,
            "--hotkey",
            args.hotkey,
            "--cache-dir",
            str(args.cache_dir.resolve()),
            "--store-dir",
            str(args.store_dir.resolve()),
            "--work-dir",
            str(args.work_dir.resolve()),
            "--selection-json",
            str(selection_path),
            "--offsets",
            args.offsets,
            "--retention-days",
            str(args.retention_days),
            "--max-run-age-hours",
            str(args.max_run_age_hours),
        ]

        if args.dry_run:
            command.append("--dry-run")

        environment = os.environ.copy()

        existing_pythonpath = environment.get(
            "PYTHONPATH",
            "",
        )

        environment["PYTHONPATH"] = (
            str(PROJECT_ROOT)
            if not existing_pythonpath
            else (
                str(PROJECT_ROOT)
                + os.pathsep
                + existing_pythonpath
            )
        )

        subprocess.run(
            command,
            check=True,
            cwd=PROJECT_ROOT,
            env=environment,
        )

        print()
        print(
            "PASS: scheduled target-cycle "
            "precompute invocation completed"
        )


if __name__ == "__main__":
    main()
