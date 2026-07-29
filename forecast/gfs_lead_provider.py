from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from forecast.gfs_provider import GFSWeatherDataProvider
from forecast.variables import (
    GFSVariableSpec,
    canonicalize_variable_name,
    convert_gfs_to_zeus_target,
    get_variable_spec,
)


class GFSLeadForecastProvider(GFSWeatherDataProvider):
    """Load real GFS forecast leads and produce hourly Zeus tensors.

    GFS cadence:
    - F000 through F120: one-hour intervals
    - F123 onward: three-hour intervals

    Missing hourly steps after F120 are linearly interpolated. Only the final
    float16 output tensor and two float32 source fields are held in memory.
    """

    MAX_SUPPORTED_FORECAST_HOUR = 384

    def __init__(
        self,
        product: str = "pgrb2.0p25",
        max_run_age_hours: int = 18,
        cache_directory: str | Path = "data/gfs_lead_cache",
    ) -> None:
        super().__init__(
            product=product,
            max_run_age_hours=max_run_age_hours,
            cache_directory=cache_directory,
        )

    @classmethod
    def source_leads(cls, maximum_hour: int) -> tuple[int, ...]:
        """Return the GFS files needed to create every hour through maximum."""

        if maximum_hour < 0:
            raise ValueError("maximum_hour cannot be negative.")

        if maximum_hour > cls.MAX_SUPPORTED_FORECAST_HOUR:
            raise ValueError(
                f"maximum_hour cannot exceed "
                f"{cls.MAX_SUPPORTED_FORECAST_HOUR}."
            )

        hourly_end = min(maximum_hour, 120)
        leads = list(range(hourly_end + 1))

        if maximum_hour > 120:
            # Include the next published three-hour lead when the requested
            # ending hour itself is not divisible by three.
            final_source_hour = ((maximum_hour + 2) // 3) * 3
            leads.extend(range(123, final_source_hour + 1, 3))

        return tuple(leads)

    @staticmethod
    def _exact_search(spec: GFSVariableSpec, lead_hour: int) -> str:
        """Select one instantaneous field, excluding solar averages."""

        suffix = "anl" if lead_hour == 0 else f"{lead_hour} hour fcst"
        return rf"{spec.search}{suffix}$"

    def _make_lead_herbie(
        self,
        cycle_time: datetime,
        lead_hour: int,
        spec: GFSVariableSpec,
    ):
        Herbie = self._import_herbie()
        is_surface_flux = spec.product == "sfluxgrb"

        kwargs = {
            "model": "gfs",
            "product": self._product_for_spec(spec),
            "fxx": lead_hour,
            "save_dir": self.cache_directory,
        }

        if is_surface_flux:
            kwargs["priority"] = ["nomads"]

        herbie = Herbie(
            self._as_naive_utc(cycle_time),
            **kwargs,
        )

        if is_surface_flux and herbie.grib and not herbie.idx:
            herbie.idx = f"{herbie.grib}.idx"
            herbie.__dict__.pop("index_as_dataframe", None)

        return herbie

    def _load_target_field(
        self,
        cycle_time: datetime,
        lead_hour: int,
        spec: GFSVariableSpec,
    ) -> np.ndarray:
        """Load one exact GFS lead and convert it to Zeus target units."""

        herbie = self._make_lead_herbie(
            cycle_time=cycle_time,
            lead_hour=lead_hour,
            spec=spec,
        )

        if not herbie.grib:
            raise FileNotFoundError(
                f"No GFS file found for {spec.era5_name} F{lead_hour:03d}."
            )

        search = self._exact_search(spec, lead_hour)
        inventory = herbie.inventory(search)

        if inventory is None or len(inventory) != 1:
            count = 0 if inventory is None else len(inventory)
            raise RuntimeError(
                f"Expected exactly one {spec.era5_name} record for "
                f"F{lead_hour:03d}; found {count}. Search={search!r}"
            )

        dataset = herbie.xarray(search, remove_grib=False)
        field = self._extract_field(dataset, spec)
        field = self._normalize_coordinates(field)
        field = convert_gfs_to_zeus_target(field, spec)

        values = np.ascontiguousarray(field.values, dtype=np.float32)

        expected_shape = (
            self.EXPECTED_LATITUDE_SIZE,
            self.EXPECTED_LONGITUDE_SIZE,
        )

        if values.shape != expected_shape:
            raise ValueError(
                f"Expected {expected_shape} for {spec.era5_name} "
                f"F{lead_hour:03d}; received {values.shape}."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{spec.era5_name} F{lead_hour:03d} contains NaN or Inf."
            )

        return values

    @staticmethod
    def _fill_interval(
        output: np.ndarray,
        previous_hour: int,
        previous_field: np.ndarray,
        next_hour: int,
        next_field: np.ndarray,
        maximum_hour: int,
        clip_nonnegative: bool,
    ) -> None:
        """Fill exact and interpolated hours between two source fields."""

        interval = next_hour - previous_hour
        if interval <= 0:
            raise ValueError("Source lead hours must be strictly increasing.")

        final_hour = min(next_hour, maximum_hour)

        for hour in range(previous_hour + 1, final_hour + 1):
            # Preserve published GFS source fields exactly at their native
            # lead hours. Interpolate only genuinely missing hourly steps.
            if hour == next_hour:
                values = next_field
            else:
                weight = np.float32(
                    (hour - previous_hour) / interval
                )
                values = previous_field + weight * (
                    next_field - previous_field
                )

            if clip_nonnegative:
                values = np.maximum(values, np.float32(0.0))

            output[hour] = values.astype(np.float16)

    def load_forecast_at_cycle(
        self,
        variable_name: str,
        maximum_hour: int,
        cycle_time: datetime,
    ) -> np.ndarray:
        """Return an hourly float16 forecast from F000 through maximum_hour."""

        canonical = canonicalize_variable_name(variable_name)
        spec = get_variable_spec(canonical)
        leads = self.source_leads(maximum_hour)

        output = np.empty(
            (
                maximum_hour + 1,
                self.EXPECTED_LATITUDE_SIZE,
                self.EXPECTED_LONGITUDE_SIZE,
            ),
            dtype=np.float16,
        )

        clip_nonnegative = (
            canonical == "surface_solar_radiation_downwards"
        )

        previous_hour = leads[0]
        previous_field = self._load_target_field(
            cycle_time=cycle_time,
            lead_hour=previous_hour,
            spec=spec,
        )

        if clip_nonnegative:
            previous_field = np.maximum(
                previous_field,
                np.float32(0.0),
            )

        output[0] = previous_field.astype(np.float16)

        for next_hour in leads[1:]:
            next_field = self._load_target_field(
                cycle_time=cycle_time,
                lead_hour=next_hour,
                spec=spec,
            )

            if clip_nonnegative:
                next_field = np.maximum(
                    next_field,
                    np.float32(0.0),
                )

            self._fill_interval(
                output=output,
                previous_hour=previous_hour,
                previous_field=previous_field,
                next_hour=next_hour,
                next_field=next_field,
                maximum_hour=maximum_hour,
                clip_nonnegative=clip_nonnegative,
            )

            previous_hour = next_hour
            previous_field = next_field

        expected_shape = (
            maximum_hour + 1,
            self.EXPECTED_LATITUDE_SIZE,
            self.EXPECTED_LONGITUDE_SIZE,
        )

        if output.shape != expected_shape:
            raise ValueError(
                f"Expected output shape {expected_shape}; "
                f"received {output.shape}."
            )

        if output.dtype != np.float16:
            raise TypeError(
                f"Expected float16 output; received {output.dtype}."
            )

        if not output.flags.c_contiguous:
            raise ValueError("Output is not C-contiguous.")

        if not np.isfinite(output).all():
            raise ValueError("Output contains NaN or Inf.")

        return output
