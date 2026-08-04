from __future__ import annotations

import os
from pathlib import Path


DEFAULT_MIN_FREE_GIB = 80.0


def free_gib(path: str | Path = ".") -> float:
    """Return free disk space in GiB for the filesystem containing ``path``."""
    usage = os.statvfs(Path(path))
    return (usage.f_bavail * usage.f_frsize) / (1024.0**3)


def assert_disk_safety(
    path: str | Path = ".",
    *,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
    reserve_for_operation_gib: float = 0.0,
) -> float:
    """Refuse to proceed unless at least ``min_free_gib`` remain free.

    If ``reserve_for_operation_gib`` is set, require
    ``free >= min_free_gib + reserve_for_operation_gib`` so the post-write
    free space still clears the gate.
    """
    available = free_gib(path)
    required = float(min_free_gib) + float(reserve_for_operation_gib)
    if available < required:
        raise RuntimeError(
            f"Disk safety gate: {available:.1f} GiB free at {Path(path).resolve()}; "
            f"need at least {required:.1f} GiB "
            f"(min_free={min_free_gib:g} + reserve={reserve_for_operation_gib:g})."
        )
    return available
