from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from forecast.gfs_bundle_provider import GFSBundleLeadProvider
from forecast.variables import get_variable_spec
from tools.precompute_gfs_bundle import (
    PGRB_VARIABLES,
    SOLAR_VARIABLE,
    parse_cycle,
    utc_text,
)
from tools.precompute_gfs_native_store import (
    LONG_MAXIMUM_HOUR,
    source_offset_hours,
)


DEFAULT_OFFSETS = (6, 12, 18, 24)


def validate_field(
    variable_name: str,
    source_lead: int,
    values: np.ndarray,
) -> None:
    if values.shape != (721, 1440):
        raise ValueError(
            f"{variable_name} F{source_lead:03d} "
            f"has shape {values.shape}; expected "
            "(721, 1440)."
        )

    if values.dtype != np.float32:
        raise TypeError(
            f"{variable_name} F{source_lead:03d} "
            f"has dtype {values.dtype}; expected "
            "float32."
        )

    if not values.flags.c_contiguous:
        raise ValueError(
            f"{variable_name} F{source_lead:03d} "
            "is not C-contiguous."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{variable_name} F{source_lead:03d} "
            "contains NaN or Inf."
        )

    if (
        variable_name == SOLAR_VARIABLE
        and float(values.min()) < 0.0
    ):
        raise ValueError(
            f"{variable_name} F{source_lead:03d} "
            "contains negative values."
        )


def probe_source_cycle(
    provider: GFSBundleLeadProvider,
    target_cycle: datetime,
    source_cycle: datetime,
) -> dict:
    offset = source_offset_hours(
        target_cycle,
        source_cycle,
    )

    first_source_lead = offset
    final_source_lead = (
        offset + LONG_MAXIMUM_HOUR
    )

    source_leads = provider.source_leads(
        final_source_lead
    )

    if first_source_lead not in source_leads:
        raise RuntimeError(
            f"Required first lead "
            f"F{first_source_lead:03d} is not "
            "a published GFS source lead."
        )

    if source_leads[-1] != final_source_lead:
        raise RuntimeError(
            f"Required final lead "
            f"F{final_source_lead:03d} is not "
            "the final selected source lead."
        )

    solar_spec = get_variable_spec(
        SOLAR_VARIABLE
    )

    checked_fields = 0

    for source_lead in (
        first_source_lead,
        final_source_lead,
    ):
        pgrb_fields = (
            provider.load_pgrb_fields_at_lead(
                cycle_time=source_cycle,
                lead_hour=source_lead,
            )
        )

        if set(pgrb_fields) != set(
            PGRB_VARIABLES
        ):
            raise RuntimeError(
                f"Combined PGRB response at "
                f"F{source_lead:03d} is incomplete."
            )

        solar = provider._load_target_field(
            cycle_time=source_cycle,
            lead_hour=source_lead,
            spec=solar_spec,
        )

        solar = np.maximum(
            solar,
            np.float32(0.0),
        )

        fields = {
            **pgrb_fields,
            SOLAR_VARIABLE: solar,
        }

        for variable_name, values in fields.items():
            validate_field(
                variable_name,
                source_lead,
                values,
            )
            checked_fields += 1

        del pgrb_fields
        del solar
        del fields
        gc.collect()

    return {
        "target_cycle_utc": utc_text(
            target_cycle
        ),
        "source_cycle_utc": utc_text(
            source_cycle
        ),
        "source_offset_hours": offset,
        "source_forecast_hour_start": (
            first_source_lead
        ),
        "source_forecast_hour_end": (
            final_source_lead
        ),
        "source_lead_count": len(
            tuple(
                lead
                for lead in source_leads
                if lead >= first_source_lead
            )
        ),
        "endpoint_fields_checked": (
            checked_fields
        ),
        "ready": True,
    }


def select_source_cycle(
    provider: GFSBundleLeadProvider,
    target_cycle: datetime,
    offsets: tuple[int, ...],
) -> tuple[dict, list[dict]]:
    attempts: list[dict] = []

    for offset in offsets:
        source_cycle = (
            target_cycle
            - timedelta(hours=offset)
        )

        print()
        print(
            f"Probing source "
            f"{utc_text(source_cycle)} "
            f"(offset {offset}h)",
            flush=True,
        )

        try:
            selection = probe_source_cycle(
                provider=provider,
                target_cycle=target_cycle,
                source_cycle=source_cycle,
            )
        except Exception as exc:
            attempt = {
                "source_cycle_utc": utc_text(
                    source_cycle
                ),
                "source_offset_hours": offset,
                "ready": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

            attempts.append(attempt)

            print(
                "  ready: False",
                flush=True,
            )
            print(
                "  reason:",
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        attempts.append(selection)

        print(
            "  ready: True",
            flush=True,
        )
        print(
            "  source range:",
            f'F{selection["source_forecast_hour_start"]:03d}-'
            f'F{selection["source_forecast_hour_end"]:03d}',
            flush=True,
        )

        return selection, attempts

    raise RuntimeError(
        "No complete GFS source cycle was found "
        f"for target {utc_text(target_cycle)}. "
        f"Attempts={json.dumps(attempts)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the newest complete GFS run "
            "that can supply a full Zeus H000-H360 "
            "forecast for a target cycle."
        )
    )

    parser.add_argument(
        "--target-cycle",
        required=True,
        type=parse_cycle,
    )

    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--offsets",
        default="6,12,18,24",
        help=(
            "Comma-separated candidate source "
            "offsets in hours."
        ),
    )

    parser.add_argument(
        "--max-run-age-hours",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--output-json",
        type=Path,
    )

    return parser


def parse_offsets(value: str) -> tuple[int, ...]:
    offsets = tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )

    if not offsets:
        raise ValueError(
            "At least one source offset is required."
        )

    if offsets != tuple(sorted(set(offsets))):
        raise ValueError(
            "Offsets must be unique and ascending."
        )

    for offset in offsets:
        if offset not in DEFAULT_OFFSETS:
            raise ValueError(
                "Supported source offsets are "
                "6, 12, 18, and 24 hours."
            )

    return offsets


def main() -> None:
    args = build_parser().parse_args()

    offsets = parse_offsets(
        args.offsets
    )

    provider = GFSBundleLeadProvider(
        cache_directory=args.cache_dir,
        max_run_age_hours=(
            args.max_run_age_hours
        ),
    )

    print("=" * 100)
    print("GFS SOURCE-CYCLE READINESS SELECTOR")
    print("=" * 100)
    print(
        "Target cycle:",
        utc_text(args.target_cycle),
    )
    print("Candidate offsets:", offsets)

    selection, attempts = select_source_cycle(
        provider=provider,
        target_cycle=args.target_cycle,
        offsets=offsets,
    )

    document = {
        "selection": selection,
        "attempts": attempts,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output_json.write_text(
            json.dumps(
                document,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    print()
    print("=" * 100)
    print("RESULT")
    print("=" * 100)
    print(
        "Selected source cycle:",
        selection["source_cycle_utc"],
    )
    print(
        "Selected source offset:",
        selection[
            "source_offset_hours"
        ],
    )
    print(
        "Selected source range:",
        f'F{selection["source_forecast_hour_start"]:03d}-'
        f'F{selection["source_forecast_hour_end"]:03d}',
    )
    print(
        "Endpoint fields checked:",
        selection[
            "endpoint_fields_checked"
        ],
    )
    print(
        "Output JSON:",
        args.output_json,
    )
    print()
    print(
        "PASS: newest ready GFS source "
        "cycle selected"
    )


if __name__ == "__main__":
    main()
