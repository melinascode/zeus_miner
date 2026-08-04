#!/usr/bin/env python3
"""Verify one immutable Zeus four-variable forecast bundle."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import blosc2
import numpy as np

from forecast.forecast_store import ForecastStore
from forecast.variables import supported_variables


def expected_state_keys() -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{variable}@{start}_{end}"
            for variable in supported_variables()
            for start, end in ((0, 48), (0, 360))
        )
    )


def parse_cycle(store: ForecastStore, value: str | None) -> datetime:
    if value is None or value == "latest":
        manifest = store.load_latest_manifest(
            expected_state_keys=expected_state_keys(),
            verify_files=False,
        )
        return store.parse_cycle_key(manifest["cycle_key"])
    try:
        return store.parse_cycle_key(value)
    except ValueError:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return store.normalize_cycle_time(parsed)


def finite_float16_chunks(raw: bytes, chunk_values: int = 10_000_000) -> bool:
    array = np.frombuffer(raw, dtype=np.float16)
    for start in range(0, array.size, chunk_values):
        if not np.isfinite(array[start : start + chunk_values]).all():
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--store-dir",
        default="data/forecast_store_v2",
    )
    parser.add_argument(
        "--cycle",
        default="latest",
        help="latest, cycle key (YYYYmmddTHHMMSSZ), or ISO timestamp",
    )
    parser.add_argument(
        "--hotkey",
        help="Expected hotkey. Defaults to the manifest hotkey.",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Decompress every artifact and scan shape/finite float16 values",
    )
    args = parser.parse_args()

    store = ForecastStore(Path(args.store_dir))
    cycle = parse_cycle(store, args.cycle)
    first_manifest = store.load_manifest(
        cycle,
        expected_state_keys=expected_state_keys(),
        verify_files=True,
    )
    hotkey = args.hotkey or first_manifest.get("hotkey")
    if not isinstance(hotkey, str) or not hotkey:
        raise ValueError("No hotkey supplied and manifest has no hotkey.")

    manifest = store.load_manifest(
        cycle,
        expected_state_keys=expected_state_keys(),
        verify_files=True,
        expected_hotkey=hotkey,
    )
    print(
        f"Verifying cycle={manifest['cycle_key']} artifacts="
        f"{len(manifest['artifacts'])} hotkey={hotkey}"
    )

    total_compressed = 0
    for state_key in expected_state_keys():
        metadata = manifest["artifacts"][state_key]
        payload = store.load_artifact(cycle, state_key, verify=True)
        total_compressed += len(payload)
        commitment = hashlib.sha256(
            payload + hotkey.encode("utf-8")
        ).hexdigest()
        if commitment != metadata["commitment_hash"]:
            raise ValueError(f"Commitment mismatch for {state_key}.")

        detail = ""
        if args.deep:
            raw = blosc2.decompress(payload)
            shape = tuple(int(value) for value in metadata["shape"])
            expected_raw_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
            if len(raw) != expected_raw_bytes:
                raise ValueError(
                    f"Raw byte length mismatch for {state_key}: "
                    f"{len(raw)} != {expected_raw_bytes}."
                )
            if not finite_float16_chunks(raw):
                raise ValueError(f"NaN/Inf detected in {state_key}.")
            detail = f", raw_bytes={len(raw)}"

        print(
            f"  PASS {state_key}: compressed_bytes={len(payload)}, "
            f"hash={commitment[:16]}...{detail}"
        )

    print(
        "PASS: exact bundle verified; "
        f"total_compressed={total_compressed / (1024**2):.1f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
