from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr

from evaluation.artifacts import DEFAULT_SPATIAL_SHAPE, SUPPORTED_HORIZONS
from forecast.variables import canonicalize_variable_name
from zeus.data.converter import get_converter


@dataclass(frozen=True)
class LoadedTruth:
    cycle_time: datetime
    variable: str
    horizon_hours: int
    tensor: torch.Tensor
    valid_times: tuple[datetime, ...]
    latitudes: np.ndarray
    longitudes: np.ndarray
    target_unit: str
    source_unit: str
    source_files: tuple[str, ...]
    source_sha256: tuple[str, ...]


class Era5TruthLoader:
    """Load validator-compatible ERA5 truth from existing local NetCDF files."""

    def __init__(
        self,
        *,
        target_latitudes: np.ndarray | None = None,
        target_longitudes: np.ndarray | None = None,
    ) -> None:
        self.target_latitudes = np.asarray(
            target_latitudes
            if target_latitudes is not None
            else np.linspace(-90.0, 90.0, DEFAULT_SPATIAL_SHAPE[0]),
            dtype=np.float64,
        )
        self.target_longitudes = np.asarray(
            target_longitudes
            if target_longitudes is not None
            else np.arange(-180.0, 180.0, 0.25),
            dtype=np.float64,
        )

    def load(
        self,
        files: Sequence[str | Path],
        *,
        variable: str,
        cycle_time: datetime,
        horizon_hours: int,
    ) -> LoadedTruth:
        if horizon_hours not in SUPPORTED_HORIZONS:
            raise ValueError(
                f"horizon_hours must be one of {SUPPORTED_HORIZONS}, "
                f"received {horizon_hours}."
            )
        paths = tuple(Path(file).resolve() for file in files)
        if not paths:
            raise ValueError("At least one ERA5 NetCDF file is required.")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"ERA5 files do not exist: {missing}")

        canonical_variable = canonicalize_variable_name(variable)
        cycle_utc = self._as_utc(cycle_time)
        dataset = xr.open_mfdataset(
            [str(path) for path in paths],
            combine="by_coords",
            engine="h5netcdf",
            compat="no_conflicts",
        )
        try:
            tensor, valid_times, source_unit = self._extract(
                dataset,
                variable=canonical_variable,
                cycle_time=cycle_utc,
                horizon_hours=horizon_hours,
            )
        finally:
            dataset.close()

        converter = get_converter(canonical_variable)
        return LoadedTruth(
            cycle_time=cycle_utc,
            variable=canonical_variable,
            horizon_hours=horizon_hours,
            tensor=tensor,
            valid_times=valid_times,
            latitudes=self.target_latitudes.copy(),
            longitudes=self.target_longitudes.copy(),
            target_unit=converter.unit,
            source_unit=source_unit,
            source_files=tuple(str(path) for path in paths),
            source_sha256=tuple(self._sha256_file(path) for path in paths),
        )

    def _extract(
        self,
        dataset: xr.Dataset,
        *,
        variable: str,
        cycle_time: datetime,
        horizon_hours: int,
    ) -> tuple[torch.Tensor, tuple[datetime, ...], str]:
        dataset = self._normalize_coordinates(dataset)
        converter = get_converter(variable)
        short_code = converter.short_code
        if short_code not in dataset.data_vars:
            raise KeyError(
                f"ERA5 dataset has no {short_code!r} field for {variable}. "
                f"Available fields: {sorted(dataset.data_vars)}"
            )

        if "valid_time" in dataset.dims:
            time_dimension = "valid_time"
        elif "time" in dataset.dims:
            time_dimension = "time"
        else:
            raise ValueError(
                "ERA5 dataset must have a valid_time or time dimension."
            )

        subset = dataset.sel(
            latitude=slice(
                float(self.target_latitudes[0]),
                float(self.target_latitudes[-1]),
            ),
            longitude=slice(
                float(self.target_longitudes[0]),
                float(self.target_longitudes[-1]),
            ),
        )
        desired_times = pd.date_range(
            start=cycle_time.replace(tzinfo=None),
            periods=horizon_hours + 1,
            freq="h",
        )
        available_times = pd.DatetimeIndex(
            pd.to_datetime(subset[time_dimension].values)
        )
        if available_times.has_duplicates:
            raise ValueError("ERA5 dataset contains duplicate valid times.")
        missing_times = desired_times.difference(available_times)
        if len(missing_times):
            rendered = [value.isoformat() for value in missing_times]
            raise ValueError(f"ERA5 dataset is missing valid times: {rendered}")

        subset = subset.sel({time_dimension: desired_times})
        selected_times = pd.DatetimeIndex(
            pd.to_datetime(subset[time_dimension].values)
        )
        if not selected_times.equals(desired_times):
            raise ValueError("ERA5 valid times are not in exact requested order.")

        latitude = np.asarray(subset.latitude.values, dtype=np.float64)
        longitude = np.asarray(subset.longitude.values, dtype=np.float64)
        self._require_exact_coordinates("latitude", latitude, self.target_latitudes)
        self._require_exact_coordinates(
            "longitude",
            longitude,
            self.target_longitudes,
        )

        data_array = subset[short_code].transpose(
            time_dimension,
            "latitude",
            "longitude",
        )
        source_unit = self._validate_source_unit(
            variable,
            data_array.attrs.get("units"),
        )
        raw = np.ascontiguousarray(data_array.values, dtype=np.float32)
        expected_shape = (
            horizon_hours + 1,
            len(self.target_latitudes),
            len(self.target_longitudes),
        )
        if raw.shape != expected_shape:
            raise ValueError(
                f"ERA5 truth shape {raw.shape} does not match {expected_shape}."
            )

        converted = converter.era5_to_target(torch.from_numpy(raw))
        tensor = torch.as_tensor(converted, dtype=torch.float32).contiguous()
        if tensor.shape != torch.Size(expected_shape):
            raise ValueError("ERA5 target conversion changed tensor shape.")
        if not torch.isfinite(tensor).all():
            raise ValueError("ERA5 truth contains NaN or Inf.")

        valid_times = tuple(
            timestamp.to_pydatetime().replace(tzinfo=timezone.utc)
            for timestamp in desired_times
        )
        return tensor, valid_times, source_unit

    @staticmethod
    def _validate_source_unit(variable: str, unit: object) -> str:
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(
                f"ERA5 {variable} has no units metadata; refusing conversion."
            )
        normalized = (
            unit.lower()
            .replace(" ", "")
            .replace("**", "^")
            .replace("−", "-")
        )
        expected = {
            "2m_temperature": {"k", "kelvin"},
            "100m_u_component_of_wind": {
                "ms^-1",
                "m/s",
                "ms-1",
            },
            "100m_v_component_of_wind": {
                "ms^-1",
                "m/s",
                "ms-1",
            },
            "surface_solar_radiation_downwards": {
                "jm^-2",
                "j/m^2",
                "jm-2",
            },
        }[variable]
        if normalized not in expected:
            raise ValueError(
                f"ERA5 {variable} units are {unit!r}; expected native CDS "
                "units before target conversion."
            )
        return unit

    @staticmethod
    def _normalize_coordinates(dataset: xr.Dataset) -> xr.Dataset:
        if "latitude" not in dataset.coords or "longitude" not in dataset.coords:
            raise ValueError("ERA5 dataset must have latitude/longitude coordinates.")
        if float(dataset.longitude.max()) > 180.0:
            dataset = dataset.assign_coords(
                longitude=(dataset.longitude.values + 180.0) % 360.0 - 180.0
            )
        if float(dataset.latitude.max()) > 90.0:
            dataset = dataset.assign_coords(
                latitude=dataset.latitude.values - 90.0
            )
        return dataset.sortby(["latitude", "longitude"])

    @staticmethod
    def _require_exact_coordinates(
        name: str,
        actual: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        if actual.shape != expected.shape or not np.allclose(
            actual,
            expected,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"ERA5 {name} coordinates do not match the Zeus grid."
            )

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
