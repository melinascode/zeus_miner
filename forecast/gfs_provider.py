from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from forecast.data_provider import WeatherDataProvider
from forecast.variables import (
    GFSVariableSpec,
    canonicalize_variable_name,
    convert_gfs_to_zeus_target,
    get_variable_spec,
)


logger = logging.getLogger(__name__)


class GFSWeatherDataProvider(WeatherDataProvider):
    """Load the four Zeus V2 variables from recent GFS analyses.

    Herbie subsets the matching GRIB record rather than intentionally loading
    every field in the product. Coordinates are normalized to the exact Zeus
    grid: latitude -90..90 ascending and longitude -180..179.75 ascending.

    This remains a persistence-model input provider. It makes the miner
    protocol-complete; it is not yet a competitive operational forecast model.
    """

    EXPECTED_LATITUDE_SIZE = 721
    EXPECTED_LONGITUDE_SIZE = 1440
    TARGET_LATITUDES = np.linspace(
        -90.0,
        90.0,
        EXPECTED_LATITUDE_SIZE,
        dtype=np.float64,
    )
    TARGET_LONGITUDES = np.arange(
        -180.0,
        180.0,
        0.25,
        dtype=np.float64,
    )

    def __init__(
        self,
        product: str = "pgrb2.0p25",
        max_run_age_hours: int = 18,
        cache_directory: str | Path = "data/gfs_cache",
    ) -> None:
        if max_run_age_hours < 0:
            raise ValueError("max_run_age_hours cannot be negative.")

        # Backwards-compatible override for the regular 0.25-degree product.
        # Solar radiation always uses the dedicated surface-flux product.
        self.product = product
        self.max_run_age_hours = max_run_age_hours
        self.cache_directory = str(cache_directory)

    def load_history(
        self,
        variable_name: str,
        history_hours: int,
    ) -> xr.DataArray:
        canonical = canonicalize_variable_name(variable_name)
        latest_cycle = self.find_common_available_cycle((canonical,))
        return self.load_history_at_cycle(
            variable_name=canonical,
            history_hours=history_hours,
            latest_cycle=latest_cycle,
        )

    def load_history_at_cycle(
        self,
        variable_name: str,
        history_hours: int,
        latest_cycle: datetime,
    ) -> xr.DataArray:
        """Load a variable history ending at an explicitly selected GFS cycle."""

        if history_hours < 1:
            raise ValueError("history_hours must be at least 1.")

        spec = get_variable_spec(variable_name)
        cycle = self._as_naive_utc(latest_cycle)
        required_cycles = max(1, int(np.ceil(history_hours / 6)))
        fields: list[xr.DataArray] = []

        for cycle_offset in range(required_cycles - 1, -1, -1):
            cycle_time = cycle - timedelta(hours=6 * cycle_offset)
            fields.append(self._load_analysis_field(cycle_time, spec))

        history = xr.concat(fields, dim="valid_time").sortby("valid_time")
        history.name = spec.era5_name
        history.attrs["source"] = spec.source_description
        history.attrs["source_product"] = self._product_for_spec(spec)
        history.attrs["latest_gfs_cycle_utc"] = cycle.isoformat()
        history.attrs["units"] = fields[-1].attrs.get("units", spec.source_units)
        history.attrs["target_conversion"] = fields[-1].attrs.get(
            "target_conversion", "identity"
        )

        self._validate_history(history)
        return history

    def find_common_available_cycle(
        self,
        variable_names: Iterable[str],
    ) -> datetime:
        """Return one recent cycle available for every requested variable.

        Using one common cycle prevents temperature, wind, and solar artifacts
        in a single commitment bundle from being initialized from different GFS
        runs merely because one product published later than another.
        """

        canonical_names = tuple(
            dict.fromkeys(canonicalize_variable_name(name) for name in variable_names)
        )
        if not canonical_names:
            raise ValueError("At least one variable is required.")

        specs = tuple(get_variable_spec(name) for name in canonical_names)
        candidate = self._latest_candidate_cycle()
        failures: list[str] = []
        fallback_count = max(1, int(np.ceil(self.max_run_age_hours / 6)))

        for offset in range(fallback_count + 1):
            cycle_time = candidate - timedelta(hours=6 * offset)
            cycle_failures: list[str] = []
            for spec in specs:
                try:
                    self._check_cycle_available(cycle_time, spec)
                except Exception as exc:  # source-specific availability errors
                    cycle_failures.append(f"{spec.era5_name}: {exc}")
            if not cycle_failures:
                return cycle_time
            failures.append(
                f"{cycle_time:%Y-%m-%d %H:%M UTC}: "
                + "; ".join(cycle_failures)
            )

        details = "\n".join(failures)
        raise RuntimeError(
            "No common usable GFS cycle was found for all Zeus variables.\n"
            f"{details}"
        )

    def _check_cycle_available(
        self,
        cycle_time: datetime,
        spec: GFSVariableSpec,
    ) -> None:
        Herbie = self._import_herbie()
        herbie = self._make_herbie(Herbie, cycle_time, spec)

        if not herbie.grib:
            raise FileNotFoundError("Herbie did not find a GRIB source.")

        inventory = herbie.inventory(spec.search)
        if inventory is None or len(inventory) == 0:
            raise FileNotFoundError(
                f"No GRIB record matched {spec.search!r} in "
                f"{self._product_for_spec(spec)}."
            )

    @staticmethod
    def _import_herbie():
        try:
            from herbie import Herbie
        except ImportError as exc:
            raise RuntimeError(
                "Herbie is not installed. Run: "
                "python -m pip install herbie-data cfgrib eccodes"
            ) from exc
        return Herbie

    def _product_for_spec(self, spec: GFSVariableSpec) -> str:
        return self.product if spec.product == "pgrb2.0p25" else spec.product

    def _make_herbie(
        self,
        Herbie,
        cycle_time: datetime,
        spec: GFSVariableSpec,
    ):
        is_surface_flux = spec.product == "sfluxgrb"
        extra_kwargs = (
            {"priority": ["nomads"]}
            if is_surface_flux
            else {}
        )

        herbie = Herbie(
            self._as_naive_utc(cycle_time),
            model="gfs",
            product=self._product_for_spec(spec),
            fxx=0,
            save_dir=self.cache_directory,
            **extra_kwargs,
        )

        # Herbie 2025.6.0 finds the corrected sflux GRIB URL but
        # does not automatically attach its valid companion index.
        # NOMADS publishes the index as "<GRIB URL>.idx".
        if is_surface_flux and herbie.grib and not herbie.idx:
            herbie.idx = f"{herbie.grib}.idx"
            herbie.__dict__.pop("index_as_dataframe", None)

        return herbie

    @staticmethod
    def _as_naive_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=None)
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _latest_candidate_cycle() -> datetime:
        # Herbie expects timezone-naive datetimes; these values represent UTC.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cycle_hour = (now.hour // 6) * 6
        current_nominal_cycle = now.replace(
            hour=cycle_hour,
            minute=0,
            second=0,
            microsecond=0,
        )

        # Start one run behind because the newest nominal run may still be
        # publishing when Zeus opens the commitment window.
        return current_nominal_cycle - timedelta(hours=6)

    def _load_analysis_field(
        self,
        cycle_time: datetime,
        spec: GFSVariableSpec,
    ) -> xr.DataArray:
        Herbie = self._import_herbie()
        herbie = self._make_herbie(Herbie, cycle_time, spec)
        dataset = herbie.xarray(spec.search, remove_grib=False)

        field = self._extract_field(dataset, spec)
        field = self._normalize_coordinates(field)
        field = convert_gfs_to_zeus_target(field, spec)

        field.name = spec.era5_name
        field.attrs["gfs_cycle_utc"] = cycle_time.isoformat()
        field.attrs["gfs_product"] = self._product_for_spec(spec)
        field.attrs["gfs_search"] = spec.search

        timestamp = pd.Timestamp(cycle_time)
        return field.expand_dims(valid_time=[timestamp])

    @staticmethod
    def _extract_field(
        dataset: xr.Dataset | list[xr.Dataset],
        spec: GFSVariableSpec,
    ) -> xr.DataArray:
        candidates = dataset if isinstance(dataset, list) else [dataset]

        for candidate in candidates:
            for variable in spec.data_variables:
                if variable in candidate.data_vars:
                    return candidate[variable].squeeze(drop=True)

        # cfgrib names can vary. An exact Herbie query returning exactly one
        # numeric field is unambiguous and safe to accept.
        numeric_fields: list[xr.DataArray] = []
        for candidate in candidates:
            for variable in candidate.data_vars:
                field = candidate[variable]
                if np.issubdtype(field.dtype, np.number):
                    numeric_fields.append(field)

        if len(numeric_fields) == 1:
            return numeric_fields[0].squeeze(drop=True)

        available = [list(candidate.data_vars) for candidate in candidates]
        raise KeyError(
            f"Could not extract {spec.era5_name} from the GFS response. "
            f"Available variables: {available}"
        )

    @classmethod
    def _normalize_coordinates(cls, data: xr.DataArray) -> xr.DataArray:
        rename_mapping: dict[str, str] = {}
        if "latitude" not in data.dims and "lat" in data.dims:
            rename_mapping["lat"] = "latitude"
        if "longitude" not in data.dims and "lon" in data.dims:
            rename_mapping["lon"] = "longitude"
        if rename_mapping:
            data = data.rename(rename_mapping)

        if "latitude" not in data.dims or "longitude" not in data.dims:
            raise ValueError(
                "GFS field must have latitude and longitude dimensions. "
                f"Received dimensions: {data.dims}."
            )

        data = data.squeeze(drop=True).transpose("latitude", "longitude")
        latitude = np.asarray(data.latitude.values, dtype=np.float64)
        longitude = np.asarray(data.longitude.values, dtype=np.float64)
        if latitude.ndim != 1 or longitude.ndim != 1:
            raise ValueError(
                "Only one-dimensional latitude/longitude coordinates are "
                "supported by this provider."
            )

        normalized_longitude = ((longitude + 180.0) % 360.0) - 180.0
        data = data.assign_coords(longitude=normalized_longitude)
        data = data.sortby("latitude").sortby("longitude")

        # Drop a duplicate meridian if a source happens to contain both 0/360.
        sorted_longitude = np.asarray(data.longitude.values)
        _, unique_indices = np.unique(sorted_longitude, return_index=True)
        if len(unique_indices) != len(sorted_longitude):
            data = data.isel(longitude=np.sort(unique_indices))

        source_lat = np.asarray(data.latitude.values, dtype=np.float64)
        source_lon = np.asarray(data.longitude.values, dtype=np.float64)
        already_exact = (
            data.sizes["latitude"] == cls.EXPECTED_LATITUDE_SIZE
            and data.sizes["longitude"] == cls.EXPECTED_LONGITUDE_SIZE
            and np.allclose(source_lat, cls.TARGET_LATITUDES, atol=1e-8)
            and np.allclose(source_lon, cls.TARGET_LONGITUDES, atol=1e-8)
        )

        if already_exact:
            normalized = data
        else:
            # sfluxgrb uses a Gaussian grid. Nearest-neighbour remapping is
            # deterministic and deliberately simple for this protocol baseline.
            normalized = data.interp(
                latitude=cls.TARGET_LATITUDES,
                longitude=cls.TARGET_LONGITUDES,
                method="nearest",
                kwargs={"fill_value": "extrapolate"},
            )

        normalized = normalized.assign_coords(
            latitude=cls.TARGET_LATITUDES,
            longitude=cls.TARGET_LONGITUDES,
        ).transpose("latitude", "longitude")
        return normalized.astype(np.float32)

    @classmethod
    def _validate_history(cls, history: xr.DataArray) -> None:
        expected_dimensions = ("valid_time", "latitude", "longitude")
        if history.dims != expected_dimensions:
            raise ValueError(
                f"Expected dimensions {expected_dimensions}, received "
                f"{history.dims}."
            )

        expected_spatial_shape = (
            cls.EXPECTED_LATITUDE_SIZE,
            cls.EXPECTED_LONGITUDE_SIZE,
        )
        actual_spatial_shape = (
            history.sizes["latitude"],
            history.sizes["longitude"],
        )
        if actual_spatial_shape != expected_spatial_shape:
            raise ValueError(
                f"Expected GFS spatial shape {expected_spatial_shape}, "
                f"received {actual_spatial_shape}."
            )

        latitude = np.asarray(history.latitude.values, dtype=np.float64)
        longitude = np.asarray(history.longitude.values, dtype=np.float64)
        if not np.allclose(latitude, cls.TARGET_LATITUDES, atol=1e-8):
            raise ValueError("Latitude coordinates do not match the Zeus grid.")
        if not np.allclose(longitude, cls.TARGET_LONGITUDES, atol=1e-8):
            raise ValueError("Longitude coordinates do not match the Zeus grid.")
        if not np.isfinite(history.values).all():
            raise ValueError("GFS history contains NaN or infinite values.")
