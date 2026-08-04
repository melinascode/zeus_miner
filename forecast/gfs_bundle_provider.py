from __future__ import annotations

from datetime import datetime
import re

import numpy as np

from forecast.gfs_lead_provider import GFSLeadForecastProvider
from forecast.variables import (
    GFSVariableSpec,
    VARIABLE_SPECS,
    convert_gfs_to_zeus_target,
)


class GFSBundleLeadProvider(GFSLeadForecastProvider):
    """Retrieve temperature and both wind components in one PGRB request."""

    PGRB_VARIABLES = (
        "2m_temperature",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
    )

    @staticmethod
    def _select_exact_record(
        herbie,
        spec: GFSVariableSpec,
        lead_hour: int,
    ) -> str:
        inventory = herbie.inventory(spec.search)

        if inventory is None or inventory.empty:
            raise RuntimeError(
                f"No GFS inventory record for {spec.era5_name} "
                f"at F{lead_hour:03d}."
            )

        search_values = inventory["search_this"].astype(str)

        if lead_hour == 0:
            candidates = inventory[
                search_values.str.endswith(":anl")
            ]

            if candidates.empty:
                candidates = inventory[
                    search_values.str.endswith(":0 hour fcst")
                ]
        else:
            candidates = inventory[
                search_values.str.endswith(
                    f":{lead_hour} hour fcst"
                )
            ]

        if len(candidates) != 1:
            available = search_values.tolist()
            raise RuntimeError(
                f"Expected one exact record for {spec.era5_name} "
                f"at F{lead_hour:03d}; found {len(candidates)}. "
                f"Available records: {available}"
            )

        return str(candidates.iloc[0]["search_this"])

    def load_pgrb_fields_at_lead(
        self,
        cycle_time: datetime,
        lead_hour: int,
    ) -> dict[str, np.ndarray]:
        """Load temperature, U wind and V wind using one Herbie xarray call."""

        if lead_hour < 0:
            raise ValueError("lead_hour cannot be negative.")

        if lead_hour > self.MAX_SUPPORTED_FORECAST_HOUR:
            raise ValueError(
                f"lead_hour cannot exceed "
                f"{self.MAX_SUPPORTED_FORECAST_HOUR}."
            )

        Herbie = self._import_herbie()

        herbie = Herbie(
            self._as_naive_utc(cycle_time),
            model="gfs",
            product="pgrb2.0p25",
            fxx=lead_hour,
            save_dir=self.cache_directory,
            priority=["aws", "nomads"],
        )

        if herbie.grib and not herbie.idx:
            herbie.idx = f"{herbie.grib}.idx"
            herbie.__dict__.pop("index_as_dataframe", None)

        exact_records: list[str] = []

        for variable_name in self.PGRB_VARIABLES:
            spec = VARIABLE_SPECS[variable_name]
            exact_records.append(
                self._select_exact_record(
                    herbie=herbie,
                    spec=spec,
                    lead_hour=lead_hour,
                )
            )

        combined_search = (
            r"^(?:"
            + "|".join(
                re.escape(record)
                for record in exact_records
            )
            + r")$"
        )

        combined_inventory = herbie.inventory(combined_search)

        if combined_inventory is None or len(combined_inventory) != 3:
            count = (
                0
                if combined_inventory is None
                else len(combined_inventory)
            )
            raise RuntimeError(
                f"Expected three combined PGRB records at "
                f"F{lead_hour:03d}; found {count}."
            )

        datasets = herbie.xarray(
            combined_search,
            remove_grib=False,
        )

        fields: dict[str, np.ndarray] = {}

        for variable_name in self.PGRB_VARIABLES:
            spec = VARIABLE_SPECS[variable_name]

            field = self._extract_field(datasets, spec)
            field = self._normalize_coordinates(field)
            field = convert_gfs_to_zeus_target(field, spec)

            values = np.ascontiguousarray(
                field.values,
                dtype=np.float32,
            )

            if values.shape != (721, 1440):
                raise ValueError(
                    f"{variable_name} at F{lead_hour:03d} has "
                    f"shape {values.shape}; expected (721, 1440)."
                )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{variable_name} at F{lead_hour:03d} "
                    "contains non-finite values."
                )

            fields[variable_name] = values

        return fields
