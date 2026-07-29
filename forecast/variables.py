from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GFSVariableSpec:
    """Map one Zeus challenge variable to a GFS GRIB field.

    ``source_representation`` describes the values returned by GFS before the
    Zeus runtime converter is applied. Temperature and wind already use ERA5
    native units. GFS DSWRF is a flux in W m-2, whereas raw ERA5 SSRD is an
    hourly energy in J m-2, so it must first be converted to the equivalent
    one-hour ERA5 accumulation.
    """

    era5_name: str
    aliases: tuple[str, ...]
    product: str
    search: str
    data_variables: tuple[str, ...]
    source_units: str
    source_description: str
    source_representation: str = "era5_native"


VARIABLE_SPECS: dict[str, GFSVariableSpec] = {
    "2m_temperature": GFSVariableSpec(
        era5_name="2m_temperature",
        aliases=("2m_temperature", "t2m", "temperature_2m"),
        product="pgrb2.0p25",
        search=r":TMP:2 m above ground:",
        data_variables=("t2m", "t"),
        source_units="K",
        source_description="GFS TMP at 2 m above ground",
    ),
    "100m_u_component_of_wind": GFSVariableSpec(
        era5_name="100m_u_component_of_wind",
        aliases=("100m_u_component_of_wind", "u100", "u100m"),
        product="pgrb2.0p25",
        search=r":UGRD:100 m above ground:",
        data_variables=("u100", "u"),
        source_units="m s**-1",
        source_description="GFS UGRD at 100 m above ground",
    ),
    "100m_v_component_of_wind": GFSVariableSpec(
        era5_name="100m_v_component_of_wind",
        aliases=("100m_v_component_of_wind", "v100", "v100m"),
        product="pgrb2.0p25",
        search=r":VGRD:100 m above ground:",
        data_variables=("v100", "v"),
        source_units="m s**-1",
        source_description="GFS VGRD at 100 m above ground",
    ),
    "surface_solar_radiation_downwards": GFSVariableSpec(
        era5_name="surface_solar_radiation_downwards",
        aliases=(
            "surface_solar_radiation_downwards",
            "ssrd",
            "dswrf",
        ),
        product="sfluxgrb",
        search=r":DSWRF:surface:",
        data_variables=("dswrf", "ssrd"),
        source_units="W m**-2",
        source_description="GFS downward short-wave radiation flux at surface",
        source_representation="hourly_mean_flux",
    ),
}


_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias.lower(): spec.era5_name
    for spec in VARIABLE_SPECS.values()
    for alias in spec.aliases
}


def canonicalize_variable_name(variable_name: str) -> str:
    """Return the canonical Zeus/ERA5 challenge variable name."""

    if not isinstance(variable_name, str) or not variable_name.strip():
        raise ValueError("Variable name must be a non-empty string.")

    key = variable_name.strip().lower()
    try:
        return _ALIAS_TO_CANONICAL[key]
    except KeyError as exc:
        supported = ", ".join(sorted(VARIABLE_SPECS))
        raise ValueError(
            f"Unsupported weather variable {variable_name!r}. "
            f"Supported variables: {supported}."
        ) from exc


def get_variable_spec(variable_name: str) -> GFSVariableSpec:
    canonical = canonicalize_variable_name(variable_name)
    return VARIABLE_SPECS[canonical]


def supported_variables() -> tuple[str, ...]:
    return tuple(VARIABLE_SPECS)


def convert_gfs_to_zeus_target(
    data: xr.DataArray,
    spec: GFSVariableSpec,
) -> xr.DataArray:
    """Convert a GFS field to the validator's target-unit convention.

    The uploaded Zeus 2.1.1 validator source calls
    ``get_converter(variable).era5_to_target(...)`` before scoring, but the
    user's source archive intentionally omitted ``zeus/data``. This function
    therefore resolves that converter dynamically from the installed Zeus
    tree at runtime.

    For DSWRF, GFS supplies W m-2. We first form the raw ERA5-equivalent
    one-hour energy (J m-2), then apply the same validator converter. If an
    older Zeus tree does not expose ``era5_to_target``, the conservative
    fallback is W m-2 for solar and native ERA5 units for temperature/wind.
    """

    source = data.astype(np.float32)
    if spec.source_representation == "hourly_mean_flux":
        era5_equivalent = source * np.float32(3600.0)
        era5_equivalent.attrs = dict(source.attrs)
        era5_equivalent.attrs["units"] = "J m**-2"
        era5_equivalent.attrs[
            "source_conversion"
        ] = "GFS DSWRF [W m-2] multiplied by 3600 s"
    else:
        era5_equivalent = source

    converter: Any | None = None
    try:
        from zeus.data.converter import get_converter

        converter = get_converter(spec.era5_name)
    except (ImportError, ModuleNotFoundError, NotImplementedError, KeyError) as exc:
        logger.warning(
            "Zeus target converter unavailable for %s; using safe baseline "
            "units (%s): %s",
            spec.era5_name,
            "W m-2" if spec.source_representation == "hourly_mean_flux" else spec.source_units,
            exc,
        )

    conversion_name = "identity"
    target_values: Any
    if converter is not None and callable(
        getattr(converter, "era5_to_target", None)
    ):
        conversion_name = f"{type(converter).__name__}.era5_to_target"
        try:
            target_values = converter.era5_to_target(era5_equivalent.values)
        except (TypeError, AttributeError):
            # Some branch-specific converters are implemented only for torch.
            try:
                import torch
            except ImportError:
                raise
            target_values = converter.era5_to_target(
                torch.from_numpy(
                    np.ascontiguousarray(era5_equivalent.values)
                )
            )
    elif spec.source_representation == "hourly_mean_flux":
        # This is also the only practical float16 representation for normal
        # daytime solar values when no branch-specific converter is available.
        conversion_name = "fallback_hourly_mean_flux_W_m-2"
        target_values = source.values
    else:
        target_values = era5_equivalent.values

    # Some converter implementations return torch tensors.
    try:
        import torch

        if isinstance(target_values, torch.Tensor):
            target_values = target_values.detach().cpu().numpy()
    except ImportError:
        pass

    values = np.asarray(target_values, dtype=np.float32)
    if values.shape != source.shape:
        raise ValueError(
            f"Target-unit converter changed {spec.era5_name} shape from "
            f"{source.shape} to {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError(
            f"Target-unit conversion produced NaN or Inf for {spec.era5_name}."
        )

    # Zeus serializes float16. Detect incompatible unit conventions before a
    # silent overflow can turn valid solar values into infinities.
    float16_limit = np.finfo(np.float16).max
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    if maximum > float16_limit:
        raise OverflowError(
            f"Converted {spec.era5_name} maximum {maximum:.3f} exceeds the "
            f"float16 limit {float16_limit:.0f}. Check the installed "
            "zeus.data.converter target-unit convention."
        )

    converted = xr.DataArray(
        values,
        coords=source.coords,
        dims=source.dims,
        name=spec.era5_name,
        attrs=dict(source.attrs),
    )
    converted.attrs["source_units"] = spec.source_units
    converted.attrs["target_conversion"] = conversion_name
    converted.attrs["units"] = getattr(converter, "target_unit", None) or getattr(
        converter, "unit", None
    ) or (
        "W m**-2"
        if conversion_name == "fallback_hourly_mean_flux_W_m-2"
        else spec.source_units
    )
    return converted
