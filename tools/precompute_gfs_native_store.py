from __future__ import annotations

import argparse
import fcntl
import gc
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from forecast.forecast_store import ForecastStore
from forecast.gfs_bundle_provider import GFSBundleLeadProvider
from forecast.variables import get_variable_spec
from tools.precompute_gfs_bundle import (
    ALL_VARIABLES,
    LONG_SHAPE,
    PGRB_VARIABLES,
    SHORT_SHAPE,
    SOLAR_VARIABLE,
    close_memmaps,
    create_memmaps,
    parse_cycle,
    utc_text,
    validate_and_measure,
)
from zeus import __version__ as zeus_version
from zeus.utils.compression import compress_prediction
from zeus.utils.hash import prediction_hash


SHORT_MAXIMUM_HOUR = 48
LONG_MAXIMUM_HOUR = 360

EXPECTED_STATE_KEYS = tuple(
    sorted(
        f"{variable_name}@0_{maximum_hour}"
        for variable_name in ALL_VARIABLES
        for maximum_hour in (
            SHORT_MAXIMUM_HOUR,
            LONG_MAXIMUM_HOUR,
        )
    )
)


def validate_cycle(value: datetime, name: str) -> None:
    if value.hour not in (0, 6, 12, 18):
        raise ValueError(
            f"{name} must be a 00/06/12/18 UTC cycle."
        )

    if any(
        (
            value.minute,
            value.second,
            value.microsecond,
        )
    ):
        raise ValueError(
            f"{name} must have zero minutes and seconds."
        )



def source_offset_hours(
    target_cycle: datetime,
    gfs_cycle: datetime,
) -> int:
    """Return the whole-hour source lead corresponding to target hour zero."""

    offset_seconds = (
        target_cycle - gfs_cycle
    ).total_seconds()

    if offset_seconds < 0:
        raise ValueError(
            "The source GFS cycle cannot be later than "
            "the target Zeus cycle."
        )

    if offset_seconds % 3600 != 0:
        raise ValueError(
            "Source/target offset must be a whole number "
            "of hours."
        )

    offset = int(offset_seconds // 3600)

    if offset % 6 != 0:
        raise ValueError(
            "Source/target cycles must differ by a "
            "multiple of six hours."
        )

    required_final_lead = (
        offset + LONG_MAXIMUM_HOUR
    )

    if (
        required_final_lead
        > GFSBundleLeadProvider.MAX_SUPPORTED_FORECAST_HOUR
    ):
        raise ValueError(
            "Source GFS cycle is too old for a complete "
            "361-hour target forecast. "
            f"offset_hours={offset}, "
            f"required_final_lead=F{required_final_lead:03d}, "
            "maximum_supported_lead="
            f"F{GFSBundleLeadProvider.MAX_SUPPORTED_FORECAST_HOUR:03d}."
        )

    return offset


def expected_shape(
    maximum_hour: int,
) -> tuple[int, int, int]:
    if maximum_hour == SHORT_MAXIMUM_HOUR:
        return SHORT_SHAPE

    if maximum_hour == LONG_MAXIMUM_HOUR:
        return LONG_SHAPE

    raise ValueError(
        f"Unsupported forecast horizon: {maximum_hour}"
    )


def requested_hours(maximum_hour: int) -> int:
    return maximum_hour + 1


def load_existing_bundle(
    store: ForecastStore,
    target_cycle: datetime,
    hotkey: str,
) -> dict | None:
    try:
        return store.load_manifest(
            target_cycle,
            expected_state_keys=EXPECTED_STATE_KEYS,
            verify_files=True,
            expected_hotkey=hotkey,
        )
    except FileNotFoundError:
        return None



def generate_aligned_long_tensors(
    provider: GFSBundleLeadProvider,
    cycle: datetime,
    outputs: dict[str, np.memmap],
    source_offset: int,
) -> tuple[int, float]:
    """Generate target H000-H360 from source F[offset]-F[offset+360]."""

    source_final_hour = (
        source_offset + LONG_MAXIMUM_HOUR
    )

    source_leads = tuple(
        lead
        for lead in provider.source_leads(
            source_final_hour
        )
        if lead >= source_offset
    )

    if not source_leads:
        raise RuntimeError(
            "No source GFS leads were selected."
        )

    if source_leads[0] != source_offset:
        raise RuntimeError(
            "The first selected source lead does not "
            f"equal F{source_offset:03d}."
        )

    if source_leads[-1] != source_final_hour:
        raise RuntimeError(
            "The final selected source lead does not "
            f"equal F{source_final_hour:03d}."
        )

    solar_spec = get_variable_spec(
        SOLAR_VARIABLE
    )

    started = time.monotonic()

    previous_source_hour = source_leads[0]
    previous_target_hour = 0

    previous_pgrb = (
        provider.load_pgrb_fields_at_lead(
            cycle_time=cycle,
            lead_hour=previous_source_hour,
        )
    )

    previous_solar = provider._load_target_field(
        cycle_time=cycle,
        lead_hour=previous_source_hour,
        spec=solar_spec,
    )

    previous_solar = np.maximum(
        previous_solar,
        np.float32(0.0),
    )

    for variable_name in PGRB_VARIABLES:
        outputs[variable_name][0] = (
            previous_pgrb[variable_name].astype(
                np.float16
            )
        )

    outputs[SOLAR_VARIABLE][0] = (
        previous_solar.astype(np.float16)
    )

    print(
        f"Loaded source F{previous_source_hour:03d} "
        "-> target H000 "
        f"(1/{len(source_leads)})",
        flush=True,
    )

    for source_number, next_source_hour in enumerate(
        source_leads[1:],
        start=2,
    ):
        next_target_hour = (
            next_source_hour - source_offset
        )

        next_pgrb = (
            provider.load_pgrb_fields_at_lead(
                cycle_time=cycle,
                lead_hour=next_source_hour,
            )
        )

        next_solar = provider._load_target_field(
            cycle_time=cycle,
            lead_hour=next_source_hour,
            spec=solar_spec,
        )

        next_solar = np.maximum(
            next_solar,
            np.float32(0.0),
        )

        for variable_name in PGRB_VARIABLES:
            provider._fill_interval(
                output=outputs[variable_name],
                previous_hour=previous_target_hour,
                previous_field=previous_pgrb[
                    variable_name
                ],
                next_hour=next_target_hour,
                next_field=next_pgrb[
                    variable_name
                ],
                maximum_hour=LONG_MAXIMUM_HOUR,
                clip_nonnegative=False,
            )

        provider._fill_interval(
            output=outputs[SOLAR_VARIABLE],
            previous_hour=previous_target_hour,
            previous_field=previous_solar,
            next_hour=next_target_hour,
            next_field=next_solar,
            maximum_hour=LONG_MAXIMUM_HOUR,
            clip_nonnegative=True,
        )

        previous_source_hour = next_source_hour
        previous_target_hour = next_target_hour
        previous_pgrb = next_pgrb
        previous_solar = next_solar

        if (
            source_number % 10 == 0
            or source_number == len(source_leads)
        ):
            print(
                f"Loaded source F{next_source_hour:03d} "
                f"-> target H{next_target_hour:03d} "
                f"({source_number}/{len(source_leads)})",
                flush=True,
            )

    for values in outputs.values():
        if hasattr(values, "flush"):
            values.flush()

    elapsed = time.monotonic() - started

    return len(source_leads), elapsed


def write_native_bundle(
    store: ForecastStore,
    target_cycle: datetime,
    gfs_cycle: datetime,
    hotkey: str,
    outputs: dict[str, np.memmap],
    source_lead_count: int,
    source_offset: int,
    generation_seconds: float,
) -> tuple[dict, float]:
    compression_seconds_total = 0.0

    source_variables = {
        variable_name: {
            "provider": "GFS",
            "gfs_cycle_utc": utc_text(gfs_cycle),
            "source_lead_count": source_lead_count,
            "source_forecast_hour_start": source_offset,
            "source_forecast_hour_end": (
                source_offset + LONG_MAXIMUM_HOUR
            ),
            "forecast_method": (
                "native GFS forecast leads with linear "
                "hourly interpolation after F120"
            ),
        }
        for variable_name in ALL_VARIABLES
    }

    with store.begin_bundle(
        target_cycle,
        EXPECTED_STATE_KEYS,
    ) as writer:
        for variable_name in ALL_VARIABLES:
            values = outputs[variable_name]

            minimum, maximum = validate_and_measure(
                values,
                variable_name,
            )

            for maximum_hour in (
                SHORT_MAXIMUM_HOUR,
                LONG_MAXIMUM_HOUR,
            ):
                state_key = (
                    f"{variable_name}@0_{maximum_hour}"
                )

                shape = expected_shape(maximum_hour)

                if maximum_hour == SHORT_MAXIMUM_HOUR:
                    tensor = np.ascontiguousarray(
                        values[: requested_hours(maximum_hour)]
                    )
                else:
                    tensor = values

                if tensor.shape != shape:
                    raise ValueError(
                        f"{state_key} has shape "
                        f"{tensor.shape}; expected {shape}."
                    )

                if tensor.dtype != np.float16:
                    raise TypeError(
                        f"{state_key} has dtype "
                        f"{tensor.dtype}; expected float16."
                    )

                if not tensor.flags.c_contiguous:
                    raise ValueError(
                        f"{state_key} is not C-contiguous."
                    )

                print(
                    f"Compressing {state_key}",
                    flush=True,
                )

                compression_started = time.monotonic()

                compressed = compress_prediction(
                    tensor
                )

                compression_seconds = (
                    time.monotonic()
                    - compression_started
                )

                compression_seconds_total += (
                    compression_seconds
                )

                commitment_hash = prediction_hash(
                    compressed,
                    hotkey,
                )

                writer.write_artifact(
                    state_key,
                    compressed,
                    shape=tensor.shape,
                    dtype=str(tensor.dtype),
                    variable=variable_name,
                    requested_hours=requested_hours(
                        maximum_hour
                    ),
                    commitment_hash=commitment_hash,
                    source_valid_time_utc=utc_text(
                        target_cycle
                    ),
                )

                print(
                    f"Stored {state_key}: "
                    f"{len(compressed):,} bytes | "
                    f"hash={commitment_hash[:16]}... | "
                    f"compression={compression_seconds:.2f}s",
                    flush=True,
                )

                del compressed

                if maximum_hour == SHORT_MAXIMUM_HOUR:
                    del tensor

                gc.collect()

        manifest = writer.finalize(
            metadata={
                "model": (
                    "gfs_native_leads_with_"
                    "linear_hourly_interpolation"
                ),
                "hotkey": hotkey,
                "fallback": False,
                "gfs_common_cycle_utc": utc_text(
                    gfs_cycle
                ),
                "source_variables": source_variables,
                "gfs_source_lead_count": (
                    source_lead_count
                ),
                "gfs_source_offset_hours": (
                    source_offset
                ),
                "gfs_source_forecast_hour_start": (
                    source_offset
                ),
                "gfs_source_forecast_hour_end": (
                    source_offset
                    + LONG_MAXIMUM_HOUR
                ),
                "precompute_generation_seconds": round(
                    generation_seconds,
                    2,
                ),
                "precompute_compression_seconds": round(
                    compression_seconds_total,
                    2,
                ),
                "precompute_total_seconds": round(
                    generation_seconds
                    + compression_seconds_total,
                    2,
                ),
                "zeus_version": zeus_version,
            }
        )

    return manifest, compression_seconds_total



def prune_completed_store(
    store: ForecastStore,
    protected_cycle_key: str,
) -> list:
    """Remove expired bundles while protecting the completed target cycle."""

    removed = store.prune(
        protect_cycle_keys=(
            protected_cycle_key,
        )
    )

    print(
        "Pruned expired bundles:",
        len(removed),
    )

    return removed


def run(args: argparse.Namespace) -> dict:
    target_cycle: datetime = args.target_cycle
    gfs_cycle: datetime = args.gfs_cycle

    validate_cycle(
        target_cycle,
        "--target-cycle",
    )
    validate_cycle(
        gfs_cycle,
        "--gfs-cycle",
    )

    source_offset = source_offset_hours(
        target_cycle,
        gfs_cycle,
    )

    store_directory = args.store_dir.resolve()
    cache_directory = args.cache_dir.resolve()
    work_root = args.work_dir.resolve()

    store_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    cache_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    work_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    store = ForecastStore(
        directory=store_directory,
        retention_days=args.retention_days,
    )

    target_key = store.cycle_key(
        target_cycle
    )

    lock_path = (
        store_directory
        / ".gfs-native-precompute.lock"
    )

    with lock_path.open("a+") as lock_handle:
        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_EX,
        )

        existing = load_existing_bundle(
            store,
            target_cycle,
            args.hotkey,
        )

        if existing is not None:
            print(
                "SKIP: complete native ForecastStore "
                f"bundle already exists for {target_key}"
            )

            prune_completed_store(
                store,
                existing["cycle_key"],
            )

            return existing

        work_directory = (
            work_root
            / f".{target_key}.work"
        )

        if work_directory.exists():
            shutil.rmtree(
                work_directory
            )

        work_directory.mkdir(
            parents=True
        )

        provider = GFSBundleLeadProvider(
            cache_directory=cache_directory,
            max_run_age_hours=(
                args.max_run_age_hours
            ),
            sflux_priority=tuple(
                getattr(
                    args,
                    "sflux_priority",
                    ["nomads"],
                )
            ),
        )

        outputs: dict[str, np.memmap] = {}

        try:
            print("=" * 100)
            print(
                "DIRECT NATIVE FORECASTSTORE "
                "GFS PRECOMPUTE"
            )
            print("=" * 100)
            print(
                "Target Zeus cycle:",
                utc_text(target_cycle),
            )
            print(
                "Source GFS cycle:",
                utc_text(gfs_cycle),
            )
            print(
                "Source forecast offset:",
                source_offset,
                "hours",
            )
            print(
                "Source forecast range:",
                f"F{source_offset:03d}-"
                f"F{source_offset + LONG_MAXIMUM_HOUR:03d}",
            )
            print(
                "Native store:",
                store_directory,
            )
            print(
                "Cache:",
                cache_directory,
            )
            print(
                "Work directory:",
                work_directory,
            )
            print()

            outputs = create_memmaps(
                work_directory
            )

            (
                source_lead_count,
                generation_seconds,
            ) = generate_aligned_long_tensors(
                provider=provider,
                cycle=gfs_cycle,
                outputs=outputs,
                source_offset=source_offset,
            )

            print()
            print(
                "Generation seconds:",
                round(
                    generation_seconds,
                    2,
                ),
            )
            print()

            (
                manifest,
                compression_seconds,
            ) = write_native_bundle(
                store=store,
                target_cycle=target_cycle,
                gfs_cycle=gfs_cycle,
                hotkey=args.hotkey,
                outputs=outputs,
                source_lead_count=(
                    source_lead_count
                ),
                source_offset=source_offset,
                generation_seconds=(
                    generation_seconds
                ),
            )

            close_memmaps(outputs)
            shutil.rmtree(
                work_directory
            )

            verified = store.load_manifest(
                target_cycle,
                expected_state_keys=(
                    EXPECTED_STATE_KEYS
                ),
                verify_files=True,
                expected_hotkey=args.hotkey,
            )

            if len(
                verified["artifacts"]
            ) != 8:
                raise RuntimeError(
                    "Native bundle does not contain "
                    "exactly eight artifacts."
                )

            prune_completed_store(
                store,
                verified["cycle_key"],
            )

            print()
            print("=" * 100)
            print("RESULT")
            print("=" * 100)
            print(
                "Target cycle:",
                verified["cycle_key"],
            )
            print(
                "Source GFS cycle:",
                verified[
                    "gfs_common_cycle_utc"
                ],
            )
            print(
                "Schema version:",
                verified["schema_version"],
            )
            print(
                "Artifacts:",
                len(
                    verified["artifacts"]
                ),
            )
            print(
                "Generation seconds:",
                round(
                    generation_seconds,
                    2,
                ),
            )
            print(
                "Compression seconds:",
                round(
                    compression_seconds,
                    2,
                ),
            )
            print(
                "Total seconds:",
                round(
                    generation_seconds
                    + compression_seconds,
                    2,
                ),
            )
            print(
                "Manifest:",
                store.bundle_path(
                    target_cycle
                )
                / store.MANIFEST_FILENAME,
            )
            print()
            print(
                "PASS: eight-artifact GFS bundle "
                "published directly to native "
                "ForecastStore"
            )

            return manifest

        finally:
            if outputs:
                close_memmaps(outputs)

            if work_directory.exists():
                shutil.rmtree(
                    work_directory,
                    ignore_errors=True,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate eight Zeus GFS artifacts "
            "directly into the schema-v2 native "
            "ForecastStore."
        )
    )

    parser.add_argument(
        "--target-cycle",
        required=True,
        type=parse_cycle,
        help=(
            "Zeus bundle cycle in "
            "YYYYMMDDTHHMMSSZ format."
        ),
    )

    parser.add_argument(
        "--gfs-cycle",
        required=True,
        type=parse_cycle,
        help=(
            "Source GFS run in "
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
        "--sflux-priority",
        default="nomads",
        help=(
            "Comma-separated Herbie sources for sfluxgrb. "
            "Use 'aws,nomads' for historical evaluation archives."
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if isinstance(args.sflux_priority, str):
        args.sflux_priority = [
            part.strip()
            for part in args.sflux_priority.split(",")
            if part.strip()
        ]
    run(args)


if __name__ == "__main__":
    main()
