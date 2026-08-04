from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from forecast.gfs_bundle_provider import GFSBundleLeadProvider
from forecast.variables import VARIABLE_SPECS, get_variable_spec
from zeus.utils.compression import compress_prediction
from zeus.utils.hash import prediction_hash


LONG_MAXIMUM_HOUR = 360
SHORT_MAXIMUM_HOUR = 48

LONG_SHAPE = (361, 721, 1440)
SHORT_SHAPE = (49, 721, 1440)

PGRB_VARIABLES = (
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
)

SOLAR_VARIABLE = "surface_solar_radiation_downwards"

ALL_VARIABLES = (
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    SOLAR_VARIABLE,
)


def parse_cycle(value: str) -> datetime:
    try:
        return datetime.strptime(
            value,
            "%Y%m%dT%H%M%SZ",
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Cycle must use YYYYMMDDTHHMMSSZ, for example "
            "20260727T000000Z."
        ) from exc


def utc_text(value: datetime) -> str:
    return value.replace(
        tzinfo=timezone.utc
    ).isoformat().replace("+00:00", "Z")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)

    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )

    with temporary_path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary_path, path)
    fsync_directory(path.parent)


def atomic_write_json(
    path: Path,
    document: dict,
) -> None:
    encoded = (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    atomic_write_bytes(path, encoded)


def validate_and_measure(
    values: np.ndarray,
    variable_name: str,
) -> tuple[float, float]:
    if values.shape != LONG_SHAPE:
        raise ValueError(
            f"{variable_name} has shape {values.shape}; "
            f"expected {LONG_SHAPE}."
        )

    if values.dtype != np.float16:
        raise TypeError(
            f"{variable_name} has dtype {values.dtype}; "
            "expected float16."
        )

    if not values.flags.c_contiguous:
        raise ValueError(
            f"{variable_name} is not C-contiguous."
        )

    minimum = float("inf")
    maximum = float("-inf")

    # Validate in small time chunks to avoid allocating one very
    # large temporary boolean tensor.
    for start in range(0, LONG_SHAPE[0], 8):
        chunk = np.asarray(
            values[start : start + 8]
        )

        if not np.isfinite(chunk).all():
            raise ValueError(
                f"{variable_name} contains NaN or Inf "
                f"near forecast hour {start}."
            )

        minimum = min(
            minimum,
            float(chunk.min()),
        )
        maximum = max(
            maximum,
            float(chunk.max()),
        )

    if variable_name == SOLAR_VARIABLE and minimum < 0.0:
        raise ValueError(
            "Solar forecast contains negative values."
        )

    return minimum, maximum


def create_memmaps(
    work_directory: Path,
) -> dict[str, np.memmap]:
    outputs: dict[str, np.memmap] = {}

    for variable_name in ALL_VARIABLES:
        path = (
            work_directory
            / f"{variable_name}.float16.memmap"
        )

        outputs[variable_name] = np.memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=LONG_SHAPE,
        )

    return outputs


def generate_long_tensors(
    provider: GFSBundleLeadProvider,
    cycle: datetime,
    outputs: dict[str, np.memmap],
) -> tuple[int, float]:
    source_leads = provider.source_leads(
        LONG_MAXIMUM_HOUR
    )

    if source_leads[0] != 0:
        raise RuntimeError(
            "The first GFS source lead is not F000."
        )

    if source_leads[-1] != LONG_MAXIMUM_HOUR:
        raise RuntimeError(
            "The final GFS source lead is not F360."
        )

    solar_spec = get_variable_spec(
        SOLAR_VARIABLE
    )

    started = time.monotonic()

    previous_hour = source_leads[0]

    previous_pgrb = (
        provider.load_pgrb_fields_at_lead(
            cycle_time=cycle,
            lead_hour=previous_hour,
        )
    )

    previous_solar = provider._load_target_field(
        cycle_time=cycle,
        lead_hour=previous_hour,
        spec=solar_spec,
    )

    previous_solar = np.maximum(
        previous_solar,
        np.float32(0.0),
    )

    for variable_name in PGRB_VARIABLES:
        outputs[variable_name][previous_hour] = (
            previous_pgrb[variable_name].astype(
                np.float16
            )
        )

    outputs[SOLAR_VARIABLE][previous_hour] = (
        previous_solar.astype(np.float16)
    )

    print(
        f"Loaded F{previous_hour:03d} "
        f"(1/{len(source_leads)})",
        flush=True,
    )

    for source_number, next_hour in enumerate(
        source_leads[1:],
        start=2,
    ):
        next_pgrb = (
            provider.load_pgrb_fields_at_lead(
                cycle_time=cycle,
                lead_hour=next_hour,
            )
        )

        next_solar = provider._load_target_field(
            cycle_time=cycle,
            lead_hour=next_hour,
            spec=solar_spec,
        )

        next_solar = np.maximum(
            next_solar,
            np.float32(0.0),
        )

        for variable_name in PGRB_VARIABLES:
            provider._fill_interval(
                output=outputs[variable_name],
                previous_hour=previous_hour,
                previous_field=previous_pgrb[
                    variable_name
                ],
                next_hour=next_hour,
                next_field=next_pgrb[
                    variable_name
                ],
                maximum_hour=LONG_MAXIMUM_HOUR,
                clip_nonnegative=False,
            )

        provider._fill_interval(
            output=outputs[SOLAR_VARIABLE],
            previous_hour=previous_hour,
            previous_field=previous_solar,
            next_hour=next_hour,
            next_field=next_solar,
            maximum_hour=LONG_MAXIMUM_HOUR,
            clip_nonnegative=True,
        )

        previous_hour = next_hour
        previous_pgrb = next_pgrb
        previous_solar = next_solar

        if (
            source_number % 10 == 0
            or source_number == len(source_leads)
        ):
            print(
                f"Loaded F{next_hour:03d} "
                f"({source_number}/"
                f"{len(source_leads)})",
                flush=True,
            )

    for values in outputs.values():
        values.flush()

    elapsed = time.monotonic() - started

    return len(source_leads), elapsed


def create_artifacts(
    staging_directory: Path,
    outputs: dict[str, np.memmap],
    hotkey: str,
) -> tuple[dict[str, dict], float]:
    artifacts: dict[str, dict] = {}
    total_compression_seconds = 0.0

    for variable_name in ALL_VARIABLES:
        values = outputs[variable_name]

        minimum, maximum = validate_and_measure(
            values,
            variable_name,
        )

        for maximum_hour, expected_shape in (
            (SHORT_MAXIMUM_HOUR, SHORT_SHAPE),
            (LONG_MAXIMUM_HOUR, LONG_SHAPE),
        ):
            artifact_key = (
                f"{variable_name}@0_{maximum_hour}"
            )

            print(
                f"Compressing {artifact_key}",
                flush=True,
            )

            if maximum_hour == SHORT_MAXIMUM_HOUR:
                tensor = np.ascontiguousarray(
                    values[:49]
                )
            else:
                tensor = values

            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{artifact_key} has shape "
                    f"{tensor.shape}; expected "
                    f"{expected_shape}."
                )

            compression_started = time.monotonic()

            compressed = compress_prediction(
                tensor
            )

            compression_seconds = (
                time.monotonic()
                - compression_started
            )

            total_compression_seconds += (
                compression_seconds
            )

            filename = f"{artifact_key}.bin"
            artifact_path = (
                staging_directory / filename
            )

            atomic_write_bytes(
                artifact_path,
                compressed,
            )

            artifacts[artifact_key] = {
                "filename": filename,
                "shape": list(expected_shape),
                "dtype": "float16",
                "compressed_bytes": len(compressed),
                "compressed_sha256": (
                    hashlib.sha256(
                        compressed
                    ).hexdigest()
                ),
                "prediction_hash": prediction_hash(
                    compressed,
                    hotkey,
                ),
                "minimum": minimum,
                "maximum": maximum,
                "compression_seconds": round(
                    compression_seconds,
                    2,
                ),
            }

            print(
                f"Stored {filename}: "
                f"{len(compressed):,} bytes",
                flush=True,
            )

            del compressed

            if maximum_hour == SHORT_MAXIMUM_HOUR:
                del tensor

            gc.collect()

    return artifacts, total_compression_seconds


def close_memmaps(
    outputs: dict[str, np.memmap],
) -> None:
    for variable_name in list(outputs):
        values = outputs.pop(variable_name)
        values.flush()
        del values

    gc.collect()


def run(args: argparse.Namespace) -> Path:
    cycle: datetime = args.cycle
    cycle_key = cycle.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    store_directory = args.store_dir.resolve()
    cache_directory = args.cache_dir.resolve()

    store_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    cache_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_path = (
        store_directory / ".precompute.lock"
    )

    with lock_path.open("a+") as lock_handle:
        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_EX,
        )

        final_directory = (
            store_directory / cycle_key
        )

        if final_directory.exists():
            manifest_path = (
                final_directory / "manifest.json"
            )

            if manifest_path.is_file():
                manifest = json.loads(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )

                if (
                    manifest.get("state")
                    == "complete"
                    and len(
                        manifest.get(
                            "artifacts",
                            {},
                        )
                    )
                    == 8
                ):
                    print(
                        "SKIP: complete bundle already "
                        f"exists at {final_directory}"
                    )
                    return final_directory

            raise FileExistsError(
                "Final directory exists but is not a "
                f"complete bundle: {final_directory}"
            )

        for stale_path in store_directory.glob(
            f".{cycle_key}.staging-*"
        ):
            shutil.rmtree(
                stale_path,
                ignore_errors=True,
            )

        staging_directory = (
            store_directory
            / f".{cycle_key}.staging-{os.getpid()}"
        )

        work_directory = (
            staging_directory / ".work"
        )

        staging_directory.mkdir()
        work_directory.mkdir()

        provider = GFSBundleLeadProvider(
            cache_directory=cache_directory,
            max_run_age_hours=args.max_run_age_hours,
        )

        outputs: dict[str, np.memmap] = {}
        published = False

        try:
            print("=" * 100)
            print("ATOMIC GFS BUNDLE PRECOMPUTE")
            print("=" * 100)
            print(
                "Cycle:",
                utc_text(cycle),
            )
            print(
                "Store:",
                final_directory,
            )
            print(
                "Cache:",
                cache_directory,
            )
            print()

            outputs = create_memmaps(
                work_directory
            )

            source_lead_count, generation_seconds = (
                generate_long_tensors(
                    provider=provider,
                    cycle=cycle,
                    outputs=outputs,
                )
            )

            print()
            print(
                "Generation seconds:",
                round(generation_seconds, 2),
            )
            print()

            artifacts, compression_seconds = (
                create_artifacts(
                    staging_directory=(
                        staging_directory
                    ),
                    outputs=outputs,
                    hotkey=args.hotkey,
                )
            )

            close_memmaps(outputs)

            shutil.rmtree(work_directory)

            manifest = {
                "schema_version": 1,
                "state": "complete",
                "model": (
                    "gfs_native_leads_with_"
                    "linear_hourly_interpolation"
                ),
                "gfs_cycle_utc": utc_text(cycle),
                "created_at_utc": (
                    datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                ),
                "hotkey": args.hotkey,
                "source_lead_count": (
                    source_lead_count
                ),
                "generation_seconds": round(
                    generation_seconds,
                    2,
                ),
                "compression_seconds": round(
                    compression_seconds,
                    2,
                ),
                "total_seconds": round(
                    generation_seconds
                    + compression_seconds,
                    2,
                ),
                "artifact_count": len(artifacts),
                "total_compressed_bytes": sum(
                    artifact[
                        "compressed_bytes"
                    ]
                    for artifact
                    in artifacts.values()
                ),
                "artifacts": artifacts,
            }

            if len(artifacts) != 8:
                raise RuntimeError(
                    f"Expected eight artifacts; "
                    f"created {len(artifacts)}."
                )

            atomic_write_json(
                staging_directory
                / "manifest.json",
                manifest,
            )

            fsync_directory(
                staging_directory
            )

            os.replace(
                staging_directory,
                final_directory,
            )

            fsync_directory(
                store_directory
            )

            published = True

            print()
            print("=" * 100)
            print("RESULT")
            print("=" * 100)
            print(
                "Artifacts:",
                len(artifacts),
            )
            print(
                "Total compressed bytes:",
                manifest[
                    "total_compressed_bytes"
                ],
            )
            print(
                "Manifest:",
                final_directory
                / "manifest.json",
            )
            print()
            print(
                "PASS: atomic eight-artifact "
                "GFS bundle published"
            )

            return final_directory

        finally:
            if outputs:
                close_memmaps(outputs)

            if (
                not published
                and staging_directory.exists()
            ):
                shutil.rmtree(
                    staging_directory,
                    ignore_errors=True,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all eight Zeus GFS artifacts "
            "in a staging directory and publish the "
            "completed bundle atomically."
        )
    )

    parser.add_argument(
        "--cycle",
        required=True,
        type=parse_cycle,
        help=(
            "GFS cycle in YYYYMMDDTHHMMSSZ format."
        ),
    )

    parser.add_argument(
        "--hotkey",
        required=True,
        help=(
            "Miner hotkey used by prediction_hash."
        ),
    )

    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Herbie cache directory.",
    )

    parser.add_argument(
        "--store-dir",
        required=True,
        type=Path,
        help=(
            "Destination root for atomic bundles."
        ),
    )

    parser.add_argument(
        "--max-run-age-hours",
        type=int,
        default=24,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
