#!/usr/bin/env python3
"""Stream-fit calibrated-GFS coefficients for locked 2024 calib cycles.

For each calibration issue:
  1. Disk gate (≥80 GiB free + reserve)
  2. Ensure ERA5 days for the 0–360 h window (Google ARCO fetch)
  3. Build production-offset hist GFS into a disposable calib store
  4. Update BiasAccumulator
  5. Delete the calib bundle and prune cache/ERA5 days not needed soon

Never reads test-cycle forecasts for the fit. Writes coefficients only at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_HOTKEY = "5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR"
MIN_FREE_GIB = 80.0
RESERVE_GIB = 8.0
PRODUCTION_OFFSET = 6


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
        "--calib-store-dir",
        default="data/evaluation/forecast_store_calib_stream",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/evaluation/gfs_cache",
    )
    parser.add_argument(
        "--era5-dir",
        default="data/evaluation/era5",
    )
    parser.add_argument(
        "--coefficients-out",
        default=(
            "data/evaluation/plans/benchmark_v1_calibrated_gfs_coefficients.json"
        ),
    )
    parser.add_argument("--hotkey", default=DEFAULT_HOTKEY)
    parser.add_argument("--assume-offset", type=int, default=PRODUCTION_OFFSET)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from status; reload accumulator checkpoint if present.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from evaluation.artifacts import ForecastArtifactReader
    from evaluation.calibration import (
        BiasAccumulator,
        assert_no_test_cycle_in_fit,
    )
    from evaluation.disk_safety import assert_disk_safety, free_gib
    from evaluation.selection import (
        calibration_issue_times,
        load_selection,
        test_cycle_times,
    )
    from evaluation.truth import Era5TruthLoader

    selection = load_selection(args.selection)
    selection_sha256 = selection["content_sha256"]
    test_keys = [
        c.strftime("%Y%m%dT%H%M%SZ") for c in test_cycle_times(selection)
    ]
    issues = list(calibration_issue_times(selection))
    issue_keys = [c.strftime("%Y%m%dT%H%M%SZ") for c in issues]
    assert_no_test_cycle_in_fit(issue_keys, test_keys)

    status_path = Path(args.status_file)
    status = _load_status(status_path, selection_sha256, issue_keys)
    status["started_at_utc"] = status.get("started_at_utc") or _now()

    checkpoint_path = Path(args.coefficients_out).with_suffix(
        ".accumulator.npz"
    )
    meta_path = Path(str(checkpoint_path) + ".meta.json")
    accumulator = BiasAccumulator()
    fitted: dict[str, list[str]] = defaultdict(list)
    if args.resume and checkpoint_path.is_file() and meta_path.is_file():
        fitted, accumulator = _load_checkpoint(
            checkpoint_path, meta_path, accumulator
        )
        print(
            f"Resumed accumulator; fitted samples="
            f"{ {k: len(v) for k, v in fitted.items()} }",
            flush=True,
        )

    Path(args.calib_store_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.calib_store_dir) / "bundles").mkdir(parents=True, exist_ok=True)
    reader = ForecastArtifactReader(
        args.calib_store_dir,
        require_complete_bundle=False,
    )
    truth_loader = Era5TruthLoader()
    variables = tuple(selection["variables"])
    processed = 0

    for index, cycle in enumerate(issues):
        if args.limit and processed >= args.limit:
            break
        cycle_key = cycle.strftime("%Y%m%dT%H%M%SZ")
        entry = status["cycles"][cycle_key]
        if entry.get("status") == "complete" and args.resume:
            # Ensure fitted lists contain this cycle if checkpoint lagged.
            continue
        if all(cycle_key in fitted[variable] for variable in variables):
            entry["status"] = "complete"
            entry["production_faithful"] = True
            _write_status(status_path, status)
            continue

        print("=" * 80, flush=True)
        print(f"CALIB [{index+1}/{len(issues)}] {cycle_key}", flush=True)
        free_before = assert_disk_safety(
            PROJECT_ROOT,
            min_free_gib=MIN_FREE_GIB,
            reserve_for_operation_gib=RESERVE_GIB,
        )
        entry["status"] = "running"
        entry["started_at_utc"] = _now()
        entry["free_gib_before"] = round(free_before, 2)
        _write_status(status_path, status)

        try:
            _ensure_era5_window(
                Path(args.era5_dir),
                variables,
                cycle,
                horizon_hours=360,
            )
            offset = int(args.assume_offset)
            source = cycle - timedelta(hours=offset)
            source_key = source.strftime("%Y%m%dT%H%M%SZ")
            entry["selection"] = {
                "source_cycle_utc": source.isoformat().replace("+00:00", "Z"),
                "source_offset_hours": offset,
                "assumed": True,
            }

            calib_store = Path(args.calib_store_dir)
            _remove_bundle(calib_store, cycle_key)
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
                str(calib_store),
                "--cache-dir",
                str(args.cache_dir),
                "--work-dir",
                f"data/evaluation/gfs_work_calib/{cycle_key}",
                "--retention-days",
                "500",
                "--max-run-age-hours",
                str(24 * 500),
                "--sflux-priority",
                "aws,nomads",
            ]
            t0 = time.time()
            completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
            build_s = round(time.time() - t0, 1)
            entry["build_seconds"] = build_s
            if completed.returncode != 0:
                raise RuntimeError(f"build exit_code={completed.returncode}")

            manifest_path = calib_store / "bundles" / cycle_key / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("gfs_source_offset_hours") != offset:
                raise RuntimeError(
                    f"offset mismatch: {manifest.get('gfs_source_offset_hours')}"
                )
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

            for variable in variables:
                if cycle_key in fitted[variable]:
                    continue
                state_key = f"{variable}@0_360"
                commitment = manifest["artifacts"][state_key]["commitment_hash"]
                artifact = reader.read(
                    cycle,
                    variable,
                    360,
                    expected_hotkey=args.hotkey,
                    expected_commitment_hash=commitment,
                    expected_manifest_sha256=manifest_sha,
                )
                truth_files = _truth_files(
                    Path(args.era5_dir), variable, cycle, 360
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

            entry["status"] = "complete"
            entry["production_faithful"] = True
            entry["failure_reason"] = None
            entry["finished_at_utc"] = _now()
            entry["free_gib_after"] = round(free_gib(PROJECT_ROOT), 2)
            processed += 1
            _write_status(status_path, status)
            _save_checkpoint(checkpoint_path, meta_path, accumulator, fitted)
            print(
                f"PASS {cycle_key} build_s={build_s} "
                f"free={entry['free_gib_after']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "failed"
            entry["failure_reason"] = f"{type(exc).__name__}: {exc}"
            entry["finished_at_utc"] = _now()
            _write_status(status_path, status)
            print(f"FAIL {cycle_key}: {entry['failure_reason']}", flush=True)
            # Continue to next cycle; freeze only if enough samples later.
        finally:
            _remove_bundle(Path(args.calib_store_dir), cycle_key)
            work = Path(f"data/evaluation/gfs_work_calib/{cycle_key}")
            if work.exists():
                shutil.rmtree(work, ignore_errors=True)
            _prune_cache(Path(args.cache_dir), cycle)
            _prune_era5_outside_lookahead(
                Path(args.era5_dir),
                issues[index : index + 8],
                variables,
                protect_test_era5=True,
            )

    assert_no_test_cycle_in_fit(fitted, test_keys)
    # Require all synoptic hours present for every variable.
    counts = {variable: len(cycles) for variable, cycles in fitted.items()}
    status["fitted_counts"] = counts
    status["finished_at_utc"] = _now()
    min_count = min(counts.values()) if counts else 0
    if min_count < 1:
        status["summary"] = {"frozen": False, "reason": "no samples", **counts}
        _write_status(status_path, status)
        raise SystemExit("No calibration samples accumulated.")

    frozen = accumulator.freeze(
        selection_sha256=selection_sha256,
        plan_id=selection["plan_id"],
        fitted_issue_cycles=dict(fitted),
    )
    destination = frozen.write(args.coefficients_out)
    status["coefficients_path"] = str(destination)
    status["coefficients_sha256"] = frozen.coefficients_sha256
    status["summary"] = {
        "frozen": True,
        "coefficients_sha256": frozen.coefficients_sha256,
        "fitted_counts": counts,
        "complete_cycles": sum(
            1 for v in status["cycles"].values() if v.get("status") == "complete"
        ),
        "failed_cycles": sum(
            1 for v in status["cycles"].values() if v.get("status") == "failed"
        ),
    }
    _write_status(status_path, status)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if meta_path.exists():
        meta_path.unlink()
    print(json.dumps(status["summary"], indent=2, sort_keys=True), flush=True)
    return 0


def _ensure_era5_window(
    era5_dir: Path,
    variables: tuple[str, ...],
    cycle: datetime,
    horizon_hours: int,
) -> None:
    start = cycle.date()
    end = (cycle + timedelta(hours=horizon_hours)).date()
    missing_days = []
    day = start
    while day <= end:
        if any(
            not (era5_dir / variable / f"era5_{day.isoformat()}.nc").is_file()
            for variable in variables
        ):
            missing_days.append(day)
        day += timedelta(days=1)
    if not missing_days:
        return
    # Fetch contiguous span covering missing days.
    fetch_start = min(missing_days)
    fetch_end = max(missing_days)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "fetch_era5_google_evaluation.py"),
        "--start-date",
        fetch_start.isoformat(),
        "--end-date",
        fetch_end.isoformat(),
        "--output-dir",
        str(era5_dir),
    ]
    for variable in variables:
        cmd.extend(["--variable", variable])
    print(f"FETCH ERA5 {fetch_start} → {fetch_end}", flush=True)
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"ERA5 fetch exit_code={completed.returncode}")


def _truth_files(era5_dir: Path, variable: str, cycle, horizon_hours: int):
    files = []
    day = cycle.date()
    end = (cycle + timedelta(hours=horizon_hours)).date()
    while day <= end:
        path = era5_dir / variable / f"era5_{day.isoformat()}.nc"
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
        day += timedelta(days=1)
    return files


def _remove_bundle(store_dir: Path, cycle_key: str) -> None:
    bundle = store_dir / "bundles" / cycle_key
    if bundle.exists():
        shutil.rmtree(bundle, ignore_errors=True)
    latest = store_dir / "latest.json"
    if latest.is_file():
        try:
            if json.loads(latest.read_text()).get("cycle_key") == cycle_key:
                latest.unlink()
        except (json.JSONDecodeError, OSError):
            pass


def _prune_cache(cache_dir: Path, cycle: datetime) -> None:
    from evaluation.disk_safety import free_gib

    if free_gib(PROJECT_ROOT) >= MIN_FREE_GIB + RESERVE_GIB + 15:
        return
    keep = {
        (cycle - timedelta(hours=h)).strftime("%Y%m%d")
        for h in (0, 6, 12, 18, 24)
    }
    root = cache_dir / "gfs"
    if not root.is_dir():
        return
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name not in keep:
            shutil.rmtree(path, ignore_errors=True)


def _prune_era5_outside_lookahead(
    era5_dir: Path,
    upcoming_cycles: list[datetime],
    variables: tuple[str, ...],
    *,
    protect_test_era5: bool,
) -> None:
    """Delete 2024 calib-only ERA5 days not needed by the next few issues.

    When protect_test_era5 is set, never delete any ERA5 day in 2025 or later
    (covers early-2025 calib truth tails and the locked test windows).
    Only 2024 days outside the upcoming lookahead may be removed under disk
    pressure.
    """
    from evaluation.disk_safety import free_gib

    if free_gib(PROJECT_ROOT) >= MIN_FREE_GIB + RESERVE_GIB + 20:
        return
    needed = set()
    for cycle in upcoming_cycles:
        day = cycle.date()
        end = (cycle + timedelta(hours=360)).date()
        while day <= end:
            needed.add(day.isoformat())
            day += timedelta(days=1)
    for variable in variables:
        var_dir = era5_dir / variable
        if not var_dir.is_dir():
            continue
        for path in var_dir.glob("era5_*.nc"):
            day = path.name[len("era5_") : -len(".nc")]
            if day in needed:
                continue
            try:
                day_date = datetime.strptime(day, "%Y-%m-%d").date()
            except ValueError:
                continue
            if protect_test_era5 and day_date.year >= 2025:
                continue
            # Only prune 2024 calib days outside lookahead.
            if day_date.year == 2024:
                path.unlink(missing_ok=True)


def _save_checkpoint(path: Path, meta_path: Path, accumulator, fitted) -> None:
    import numpy as np

    payload = {}
    for variable, hours in accumulator._numerators.items():
        for hour, arr in hours.items():
            payload[f"num/{variable}/{hour}"] = arr
            payload[f"den/{variable}/{hour}"] = accumulator._denominators[
                variable
            ][hour]
            payload[f"count/{variable}/{hour}"] = np.asarray(
                [accumulator._counts[variable][hour]], dtype=np.int64
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    meta_path.write_text(
        json.dumps({"fitted": fitted}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_checkpoint(path: Path, meta_path: Path, accumulator):
    import numpy as np

    data = np.load(path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fitted = defaultdict(list, {k: list(v) for k, v in meta["fitted"].items()})
    for key in data.files:
        kind, variable, hour_s = key.split("/", 2)
        hour = int(hour_s)
        if kind == "num":
            accumulator._numerators[variable][hour] = data[key].astype(
                np.float64
            )
        elif kind == "den":
            accumulator._denominators[variable][hour] = data[key].astype(
                np.float64
            )
        elif kind == "count":
            accumulator._counts[variable][hour] = int(data[key][0])
    return fitted, accumulator


def _load_status(path: Path, selection_sha256: str, issue_keys: list[str]) -> dict:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("selection_sha256") not in (None, selection_sha256):
            raise SystemExit("Calib status bound to a different selection hash.")
        payload.setdefault("cycles", {})
        for key in issue_keys:
            payload["cycles"].setdefault(
                key,
                {
                    "status": "pending",
                    "failure_reason": None,
                    "production_faithful": False,
                },
            )
        return payload
    return {
        "schema_version": 1,
        "plan_id": "benchmark_v1",
        "selection_sha256": selection_sha256,
        "mode": "streaming_calibration",
        "assume_offset": PRODUCTION_OFFSET,
        "cycles": {
            key: {
                "status": "pending",
                "failure_reason": None,
                "production_faithful": False,
            }
            for key in issue_keys
        },
    }


def _write_status(path: Path, payload: dict) -> None:
    path = Path(path)
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
