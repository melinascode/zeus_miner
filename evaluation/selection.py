from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_SELECTION_PATH = Path(
    "data/evaluation/plans/benchmark_v1_selection.json"
)
DEFAULT_REGISTRY_PATH = Path(
    "data/evaluation/plans/benchmark_v1_registry.json"
)
ALLOWED_STATUSES = frozenset(
    {"pending", "complete", "incomplete", "failed"}
)


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def content_sha256(
    payload: dict[str, Any],
    digest_key: str = "content_sha256",
) -> str:
    without = {
        key: value
        for key, value in payload.items()
        if key != digest_key
    }
    return hashlib.sha256(canonical_json_bytes(without)).hexdigest()


def load_selection(
    path: str | Path = DEFAULT_SELECTION_PATH,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    selection_path = Path(path)
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    digest = content_sha256(payload)
    recorded = payload.get("content_sha256")
    if recorded != digest:
        raise ValueError(
            f"Selection content_sha256 mismatch for {selection_path}: "
            f"recorded={recorded} computed={digest}"
        )
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"Selection hash {digest} does not match expected "
            f"{expected_sha256}."
        )
    validate_selection(payload)
    return payload


def validate_selection(payload: dict[str, Any]) -> None:
    if payload.get("plan_id") != "benchmark_v1":
        raise ValueError("Unexpected plan_id for benchmark_v1 selection.")
    cycles = payload.get("test_cycles")
    if not isinstance(cycles, list) or len(cycles) < 30:
        raise ValueError("Selection must contain at least 30 test cycles.")
    cycle_times = [_parse_cycle(item["cycle"]) for item in cycles]
    if len(set(cycle_times)) != len(cycle_times):
        raise ValueError("Selection contains duplicate test cycles.")
    hours = {cycle.hour for cycle in cycle_times}
    if hours != {0, 6, 12, 18}:
        raise ValueError(
            "Selection must include all synoptic hours 00/06/12/18."
        )
    for previous, current in zip(cycle_times, cycle_times[1:]):
        spacing = (current - previous).total_seconds() / 3600.0
        if spacing <= 360:
            raise ValueError(
                f"Test cycles {previous.isoformat()} and "
                f"{current.isoformat()} are not independent for 360 h."
            )

    calibration = payload["calibration"]
    calib_start = _parse_cycle(calibration["issue_start"])
    calib_end = _parse_cycle(calibration["issue_end"])
    truth_end = _parse_cycle(calibration["truth_end"])
    first_test = cycle_times[0]
    if calib_end >= first_test:
        raise ValueError("Calibration issue period overlaps test issues.")
    if truth_end >= first_test:
        raise ValueError("Calibration truth window overlaps test truth.")
    expected_truth_end = calib_end + timedelta(
        hours=int(calibration["long_horizon_hours"])
    )
    if truth_end != expected_truth_end:
        raise ValueError(
            "calibration.truth_end does not match issue_end + long horizon."
        )


def assert_selection_unchanged(
    path: str | Path,
    expected_sha256: str,
) -> None:
    load_selection(path, expected_sha256=expected_sha256)


def load_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH,
    *,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if selection is not None:
        digest = selection["content_sha256"]
        if payload.get("selection_sha256") != digest:
            raise ValueError(
                "Registry selection_sha256 does not match locked selection."
            )
        expected = {item["cycle"] for item in selection["test_cycles"]}
        actual = set(payload.get("cycles", {}))
        if actual != expected:
            raise ValueError(
                "Registry cycle keys do not match locked selection cycles."
            )
    return payload


def record_cycle_status(
    registry_path: str | Path,
    cycle: str | datetime,
    status: str,
    *,
    failure_reason: str | None = None,
    selection_sha256: str | None = None,
) -> dict[str, Any]:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Unsupported cycle status: {status}")
    cycle_key = (
        cycle
        if isinstance(cycle, str)
        else cycle.strftime("%Y%m%dT%H%M%SZ")
    )
    path = Path(registry_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        selection_sha256 is not None
        and payload.get("selection_sha256") != selection_sha256
    ):
        raise ValueError(
            "Refusing to update registry bound to a different selection hash."
        )
    if cycle_key not in payload.get("cycles", {}):
        raise KeyError(
            f"Cycle {cycle_key} is not in the locked selection registry."
        )
    payload["cycles"][cycle_key] = {
        "status": status,
        "failure_reason": failure_reason,
        "updated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    _atomic_write_json(path, payload)
    return payload


def calibration_issue_times(selection: dict[str, Any]) -> tuple[datetime, ...]:
    calibration = selection["calibration"]
    start = _parse_cycle(calibration["issue_start"])
    end = _parse_cycle(calibration["issue_end"])
    step = timedelta(hours=int(calibration["issue_step_hours"]))
    issues: list[datetime] = []
    current = start
    while current <= end:
        issues.append(current)
        current += step
    expected = int(calibration["n_issue_cycles"])
    if len(issues) != expected:
        raise ValueError(
            f"Expected {expected} calibration issues, built {len(issues)}."
        )
    return tuple(issues)


def test_cycle_times(selection: dict[str, Any]) -> tuple[datetime, ...]:
    return tuple(
        _parse_cycle(item["cycle"]) for item in selection["test_cycles"]
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_cycle(value: str) -> datetime:
    parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    if parsed.hour % 6 != 0 or parsed.minute or parsed.second:
        raise ValueError(f"Cycle must be an exact UTC 6-hour boundary: {value}")
    return parsed
