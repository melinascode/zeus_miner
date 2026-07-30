from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_VARIABLES = (
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
)
SUPPORTED_HORIZONS = (48, 360)


@dataclass(frozen=True)
class BacktestCase:
    cycle_time: datetime
    variable: str
    horizon_hours: int
    truth_files: tuple[str, ...]
    commitment_hash: str
    manifest_sha256: str

    @property
    def identity(self) -> tuple[datetime, str, int]:
        return self.cycle_time, self.variable, self.horizon_hours

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BacktestCase":
        cycle_time = _parse_cycle(value["cycle"])
        variable = str(value["variable"])
        horizon_hours = int(value["horizon"])
        truth_files = tuple(str(path) for path in value["truth_files"])
        commitment_hash = str(value["commitment_hash"])
        manifest_sha256 = str(value["manifest_sha256"])
        if variable not in SUPPORTED_VARIABLES:
            raise ValueError(f"Unsupported backtest variable: {variable}")
        if horizon_hours not in SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported backtest horizon: {horizon_hours}")
        if not truth_files:
            raise ValueError("Every backtest case needs truth_files.")
        if len(commitment_hash) != 64:
            raise ValueError("Every backtest case needs a SHA-256 commitment_hash.")
        if len(manifest_sha256) != 64:
            raise ValueError("Every backtest case needs a manifest_sha256.")
        return cls(
            cycle_time=cycle_time,
            variable=variable,
            horizon_hours=horizon_hours,
            truth_files=truth_files,
            commitment_hash=commitment_hash,
            manifest_sha256=manifest_sha256,
        )


def load_backtest_plan(path: str | Path) -> tuple[BacktestCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Backtest plan must contain a non-empty cases list.")
    return tuple(BacktestCase.from_dict(value) for value in raw_cases)


def validate_backtest_plan(
    cases: Iterable[BacktestCase],
    *,
    minimum_cycles: int = 1,
    require_full_matrix: bool = False,
    require_independent: bool = False,
) -> tuple[BacktestCase, ...]:
    validated = tuple(cases)
    identities = [case.identity for case in validated]
    if len(identities) != len(set(identities)):
        raise ValueError("Backtest plan contains duplicate cases.")

    cycles = sorted({case.cycle_time for case in validated})
    if len(cycles) < minimum_cycles:
        raise ValueError(
            f"Backtest has {len(cycles)} unique cycles; "
            f"{minimum_cycles} are required."
        )

    if require_full_matrix:
        expected = {
            (variable, horizon)
            for variable in SUPPORTED_VARIABLES
            for horizon in SUPPORTED_HORIZONS
        }
        for cycle in cycles:
            actual = {
                (case.variable, case.horizon_hours)
                for case in validated
                if case.cycle_time == cycle
            }
            if actual != expected:
                raise ValueError(
                    f"Cycle {cycle.isoformat()} does not contain the exact "
                    f"four-variable/two-horizon matrix."
                )

    if require_independent and len(cycles) > 1:
        maximum_horizon = max(case.horizon_hours for case in validated)
        for previous, current in zip(cycles, cycles[1:]):
            separation_hours = (
                current - previous
            ).total_seconds() / 3600.0
            if separation_hours <= maximum_horizon:
                raise ValueError(
                    f"Cycles {previous.isoformat()} and "
                    f"{current.isoformat()} overlap for the "
                    f"{maximum_horizon} h horizon."
                )

    return tuple(
        sorted(
            validated,
            key=lambda case: (
                case.cycle_time,
                case.variable,
                case.horizon_hours,
            ),
        )
    )


def summarize_results(
    results: Iterable[dict[str, Any]],
    *,
    incomplete: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result_list = list(results)
    incomplete_list = list(incomplete or ())
    candidates = ("persistence", "raw_gfs", "calibrated_gfs")
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for result in result_list:
        case = result["case"]
        for candidate in candidates:
            if candidate not in result["metrics"]:
                continue
            grouped.setdefault(
                (
                    case["variable"],
                    case["horizon_hours"],
                    candidate,
                ),
                [],
            ).append(result["metrics"][candidate])

    summaries = []
    for (variable, horizon, candidate), rows in sorted(grouped.items()):
        summaries.append(
            {
                "variable": variable,
                "horizon_hours": horizon,
                "candidate": candidate,
                "evaluations": len(rows),
                "mean_rmse": statistics.fmean(row["rmse"] for row in rows),
                "median_rmse": statistics.median(row["rmse"] for row in rows),
                "mean_mae": statistics.fmean(row["mae"] for row in rows),
                "median_mae": statistics.median(row["mae"] for row in rows),
                "mean_combined_error": statistics.fmean(
                    row["combined_error"] for row in rows
                ),
                "median_combined_error": statistics.median(
                    row["combined_error"] for row in rows
                ),
            }
        )

    unique_cycles = sorted(
        {result["case"]["cycle_key"] for result in result_list}
    )
    return {
        "schema_version": 2 if any(
            "calibrated_gfs" in result["metrics"] for result in result_list
        )
        else 1,
        "unique_cycles": len(unique_cycles),
        "cycle_keys": unique_cycles,
        "evaluations": len(result_list),
        "incomplete_or_failed": incomplete_list,
        "incomplete_or_failed_count": len(incomplete_list),
        "raw_gfs_wins": sum(
            bool(result["metrics"]["comparison"]["raw_gfs_wins"])
            for result in result_list
        ),
        "calibrated_gfs_wins_vs_raw_gfs": sum(
            bool(
                result["metrics"]["comparison"].get(
                    "calibrated_gfs_wins_vs_raw_gfs"
                )
            )
            for result in result_list
            if "calibrated_gfs" in result["metrics"]
        ),
        "groups": summaries,
    }


def _parse_cycle(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
    if (
        parsed.minute != 0
        or parsed.second != 0
        or parsed.microsecond != 0
        or parsed.hour % 6 != 0
    ):
        raise ValueError(
            f"Backtest cycle must be an exact UTC 6-hour boundary: {value}"
        )
    return parsed
