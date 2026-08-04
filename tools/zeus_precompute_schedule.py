from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone


NOMINAL_RUN_HOURS = (5, 11, 17, 23)
NOMINAL_RUN_MINUTE = 45

SOURCE_OFFSETS = (6, 12, 18, 24)

COMMIT_OPEN_OFFSET = timedelta(minutes=30)
COMMIT_DEADLINE_OFFSET = timedelta(minutes=45)

DEFAULT_MAX_LATENESS = timedelta(minutes=30)


def parse_utc(value: str) -> datetime:
    parsed = datetime.strptime(
        value,
        "%Y%m%dT%H%M%SZ",
    )

    return parsed.replace(
        tzinfo=timezone.utc,
    )


def utc_text(value: datetime) -> str:
    return value.astimezone(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def cycle_key(value: datetime) -> str:
    return value.astimezone(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")


def nominal_slots_near(
    now: datetime,
) -> tuple[datetime, ...]:
    now = now.astimezone(
        timezone.utc
    )

    dates = (
        now.date() - timedelta(days=1),
        now.date(),
    )

    slots: list[datetime] = []

    for date_value in dates:
        for hour in NOMINAL_RUN_HOURS:
            slots.append(
                datetime(
                    date_value.year,
                    date_value.month,
                    date_value.day,
                    hour,
                    NOMINAL_RUN_MINUTE,
                    tzinfo=timezone.utc,
                )
            )

    return tuple(sorted(slots))


def resolve_schedule(
    now: datetime,
    max_lateness: timedelta = (
        DEFAULT_MAX_LATENESS
    ),
) -> dict:
    now = now.astimezone(
        timezone.utc
    )

    eligible_slots = tuple(
        slot
        for slot in nominal_slots_near(now)
        if slot <= now
    )

    if not eligible_slots:
        raise RuntimeError(
            "No nominal precompute slot could "
            "be resolved."
        )

    nominal_run = eligible_slots[-1]
    lateness = now - nominal_run

    if lateness > max_lateness:
        raise RuntimeError(
            "Scheduled precompute invocation is "
            "too late. "
            f"nominal_run={utc_text(nominal_run)}, "
            f"now={utc_text(now)}, "
            f"lateness_minutes="
            f"{lateness.total_seconds() / 60:g}, "
            f"maximum_minutes="
            f"{max_lateness.total_seconds() / 60:g}."
        )

    target_cycle = (
        nominal_run
        + timedelta(minutes=15)
    )

    commit_open = (
        target_cycle
        + COMMIT_OPEN_OFFSET
    )

    commit_deadline = (
        target_cycle
        + COMMIT_DEADLINE_OFFSET
    )

    source_candidates = [
        {
            "offset_hours": offset,
            "source_cycle_utc": utc_text(
                target_cycle
                - timedelta(hours=offset)
            ),
        }
        for offset in SOURCE_OFFSETS
    ]

    return {
        "now_utc": utc_text(now),
        "nominal_run_utc": utc_text(
            nominal_run
        ),
        "lateness_seconds": int(
            lateness.total_seconds()
        ),
        "target_cycle_utc": utc_text(
            target_cycle
        ),
        "target_cycle_key": cycle_key(
            target_cycle
        ),
        "commit_window_open_utc": utc_text(
            commit_open
        ),
        "commit_deadline_utc": utc_text(
            commit_deadline
        ),
        "minutes_nominal_to_window_open": int(
            (
                commit_open
                - nominal_run
            ).total_seconds()
            // 60
        ),
        "minutes_nominal_to_deadline": int(
            (
                commit_deadline
                - nominal_run
            ).total_seconds()
            // 60
        ),
        "source_candidates": (
            source_candidates
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the Zeus target cycle for "
            "a scheduled GFS precompute invocation."
        )
    )

    parser.add_argument(
        "--now",
        type=parse_utc,
        help=(
            "Override current UTC time using "
            "YYYYMMDDTHHMMSSZ."
        ),
    )

    parser.add_argument(
        "--max-lateness-minutes",
        type=int,
        default=30,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    now = (
        args.now
        if args.now is not None
        else datetime.now(timezone.utc)
    )

    result = resolve_schedule(
        now,
        max_lateness=timedelta(
            minutes=args.max_lateness_minutes
        ),
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
