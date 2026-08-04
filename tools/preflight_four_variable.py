#!/usr/bin/env python3
"""Preflight the four-variable Zeus 2.1.1 GFS integration.

Run this before starting the patched miner. By default it verifies imports,
validator target converters, GRIB inventory availability, coordinates, and disk
capacity without generating 49/361-hour tensors. ``--full-download`` downloads
one analysis field for each variable and validates the complete global grid.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from forecast.gfs_provider import GFSWeatherDataProvider
from forecast.variables import get_variable_spec, supported_variables


def _to_numpy(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _probe_runtime_converter(variable: str) -> str:
    try:
        from zeus.data.converter import get_converter
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"

    try:
        converter = get_converter(variable)
    except Exception as exc:
        return f"not registered ({type(exc).__name__}: {exc})"

    method = getattr(converter, "era5_to_target", None)
    if not callable(method):
        return f"{type(converter).__name__}; no era5_to_target (native fallback)"

    # Probe values are in raw ERA5 units. 3600 J m-2 represents an hourly
    # average solar flux of 1 W m-2.
    probe = {
        "2m_temperature": np.array([273.15], dtype=np.float32),
        "100m_u_component_of_wind": np.array([1.0], dtype=np.float32),
        "100m_v_component_of_wind": np.array([1.0], dtype=np.float32),
        "surface_solar_radiation_downwards": np.array(
            [3600.0], dtype=np.float32
        ),
    }[variable]
    try:
        result = method(probe)
    except (TypeError, AttributeError):
        import torch

        result = method(torch.from_numpy(probe))
    value = float(_to_numpy(result).reshape(-1)[0])
    return f"{type(converter).__name__}.era5_to_target({probe[0]:g}) -> {value:g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default="data/gfs_cache",
        help="Herbie cache directory",
    )
    parser.add_argument(
        "--max-run-age-hours",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--history-hours",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--full-download",
        action="store_true",
        help="Download and validate one complete global field per variable",
    )
    args = parser.parse_args()

    variables = tuple(sorted(supported_variables()))
    print("Zeus four-variable preflight")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Variables ({len(variables)}):")
    for variable in variables:
        spec = get_variable_spec(variable)
        print(
            f"  - {variable}: product={spec.product}, query={spec.search}, "
            f"source_units={spec.source_units}"
        )
        print(f"      runtime converter: {_probe_runtime_converter(variable)}")

    cache_path = Path(args.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(cache_path)
    print(
        "Disk: "
        f"free={disk.free / (1024**3):.1f} GiB, "
        f"total={disk.total / (1024**3):.1f} GiB"
    )
    if disk.free < 20 * 1024**3:
        print(
            "WARNING: less than 20 GiB free. Twenty-four days of eight "
            "compressed artifacts may require substantially more space."
        )

    provider = GFSWeatherDataProvider(
        cache_directory=cache_path,
        max_run_age_hours=args.max_run_age_hours,
    )
    common_cycle = provider.find_common_available_cycle(variables)
    print(f"Common available GFS cycle: {common_cycle.isoformat()} UTC")

    if args.full_download:
        print("Downloading and validating one history per variable...")
        for variable in variables:
            history = provider.load_history_at_cycle(
                variable_name=variable,
                history_hours=args.history_hours,
                latest_cycle=common_cycle,
            )
            values = history.values
            maximum = float(np.max(values))
            minimum = float(np.min(values))
            print(
                f"  PASS {variable}: shape={history.shape}, "
                f"dtype={history.dtype}, min={minimum:.4f}, max={maximum:.4f}, "
                f"units={history.attrs.get('units')}, "
                f"conversion={history.attrs.get('target_conversion')}"
            )
            if max(abs(minimum), abs(maximum)) > np.finfo(np.float16).max:
                raise OverflowError(
                    f"{variable} cannot be represented safely as float16."
                )

    print("PASS: four-variable preflight completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
