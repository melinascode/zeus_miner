"""Region masks and geographic weighting for lat-lon grids."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from zeus.validator.constants import (
    EUROPE_WEIGHT,
    EUROPE_LATITUDE_RANGE,
    EUROPE_LONGITUDE_RANGE,
    GERMANY_WEIGHT,
    GERMANY_LATITUDE_RANGE,
    GERMANY_LONGITUDE_RANGE,
    SOLAR_SCALARS_PATH,
    TEMPERATURE_SCALARS_PATH,
    WIND_SCALARS_PATH,
)


@dataclass(frozen=True)
class RegionConfig:
    name: str
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    weight: float

EUROPE_REGION_CONFIG: RegionConfig = RegionConfig(
        name="Europe",
        lat_range=EUROPE_LATITUDE_RANGE,
        lon_range=EUROPE_LONGITUDE_RANGE,
        weight=EUROPE_WEIGHT,
    )
GERMANY_REGION_CONFIG: RegionConfig = RegionConfig(
        name="Germany",
        lat_range=GERMANY_LATITUDE_RANGE,
        lon_range=GERMANY_LONGITUDE_RANGE,
        weight=GERMANY_WEIGHT,
    )
# TODO: Remove this once we have evaluated all the challenges before the update
OLD_REGION_CONFIGS: list[RegionConfig] = [
    EUROPE_REGION_CONFIG,
]

REGION_CONFIGS: list[RegionConfig] = [
    EUROPE_REGION_CONFIG,
    GERMANY_REGION_CONFIG,
]


def region_mask_for_grid(
    grid: torch.Tensor,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> torch.Tensor:
    """
    Return a mask of shape (n_lat, n_lon) with 1 inside the axis-aligned lat/lon box, 0 otherwise.
    grid must have shape (n_lat, n_lon, 2) with grid[..., 0] = lat, grid[..., 1] = lon.
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    in_lat = (grid[..., 0] >= lat_min) & (grid[..., 0] <= lat_max)
    in_lon = (grid[..., 1] >= lon_min) & (grid[..., 1] <= lon_max)
    return (in_lat & in_lon).to(torch.float32)



# Tuesday incentive-mass split (box-only interim; capacity maps land later).
GERMANY_MASS_SHARE = 0.40
REST_EUROPE_MASS_SHARE = 0.40
REST_OF_WORLD_MASS_SHARE = 0.20


def region_masks_for_grid(grid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Germany ⊂ Europe axis-aligned boxes used by the current validator."""
    europe = region_mask_for_grid(
        grid, EUROPE_LATITUDE_RANGE, EUROPE_LONGITUDE_RANGE
    )
    germany = region_mask_for_grid(
        grid, GERMANY_LATITUDE_RANGE, GERMANY_LONGITUDE_RANGE
    )
    rest_europe = (europe * (1.0 - germany)).clamp(min=0.0, max=1.0)
    rest_of_world = (1.0 - europe).clamp(min=0.0, max=1.0)
    return {
        "germany": germany,
        "rest_europe": rest_europe,
        "europe": europe,
        "rest_of_world": rest_of_world,
    }


def build_mass_share_weights(
    grid: torch.Tensor,
    latitude_weights: torch.Tensor,
    *,
    germany_share: float = GERMANY_MASS_SHARE,
    rest_europe_share: float = REST_EUROPE_MASS_SHARE,
    rest_of_world_share: float = REST_OF_WORLD_MASS_SHARE,
) -> torch.Tensor:
    """Cosine-latitude weights rescaled so region masses match incentive shares.

    Each cell keeps its relative cosine-latitude weight *inside* its box; the
    three boxes are then scaled so their totals are 40% / 40% / 20%. This is
    the Tuesday policy with uniform-inside-box proxies (no wind/solar/pop maps).
    """
    if latitude_weights.ndim != 1 or latitude_weights.shape[0] != grid.shape[0]:
        raise ValueError(
            f"latitude_weights must have shape ({grid.shape[0]},), "
            f"got {tuple(latitude_weights.shape)}"
        )
    masks = region_masks_for_grid(grid)
    cosine = latitude_weights.to(dtype=torch.float32, device=grid.device)[:, None]
    parts = [
        (masks["germany"], germany_share),
        (masks["rest_europe"], rest_europe_share),
        (masks["rest_of_world"], rest_of_world_share),
    ]
    weights = torch.zeros(grid.shape[:-1], dtype=torch.float32, device=grid.device)
    for mask, share in parts:
        raw = cosine * mask
        mass = raw.sum().clamp_min(1e-12)
        weights = weights + raw * (share / mass)
    return weights


def build_geographic_weights(
    grid: torch.Tensor,
    configs: list[RegionConfig] | None = None,
    default_weight: float = 1.0,
) -> torch.Tensor:
    """
    Per-cell multipliers: start at default_weight. For each region in order, cells inside that
    region's box are set to max(current cell, region.weight); other cells are unchanged.
    With [Europe, Germany] and Germany ⊂ Europe, Germany gets GERMANY_WEIGHT and the rest of
    Europe gets EUROPE_WEIGHT (not GERMANY_WEIGHT).
    """
    if configs is None:
        configs = REGION_CONFIGS
    weights = torch.full(
        grid.shape[:-1],
        default_weight,
        dtype=torch.float32,
        device=grid.device,
    )
    for region in configs:
        mask = region_mask_for_grid(grid, region.lat_range, region.lon_range)
        max_weight = torch.maximum(weights, torch.tensor(region.weight, device=weights.device))
        weights = torch.where(mask == 1, max_weight, weights)
    return weights


@lru_cache(maxsize=8)
def load_scalar_npz(path: str) -> dict[str, np.ndarray]:
    """Load an official Zeus scalar map. Keys: scalars (721, 1440), lats, lons."""
    payload_path = Path(path)
    with np.load(payload_path) as data:
        scalars = np.ascontiguousarray(data["scalars"])
        lats = np.ascontiguousarray(data["lats"], dtype=np.float64)
        lons = np.ascontiguousarray(data["lons"], dtype=np.float64)
    if scalars.shape != (len(lats), len(lons)):
        raise ValueError(
            f"{payload_path} scalars {scalars.shape} do not match "
            f"lats {lats.shape} / lons {lons.shape}"
        )
    return {"scalars": scalars, "lats": lats, "lons": lons}


def geographic_scalar_for_variable(variable: str) -> torch.Tensor:
    """Full-globe (721, 1440) capacity/population scalar used after 2026-08-25 18:00 UTC."""
    paths = {
        "2m_temperature": TEMPERATURE_SCALARS_PATH,
        "100m_u_component_of_wind": WIND_SCALARS_PATH,
        "100m_v_component_of_wind": WIND_SCALARS_PATH,
        "surface_solar_radiation_downwards": SOLAR_SCALARS_PATH,
    }
    if variable not in paths:
        raise KeyError(f"No capacity scalar map for variable {variable!r}")
    payload = load_scalar_npz(str(paths[variable]))
    return torch.from_numpy(payload["scalars"].astype(np.float32, copy=False))


if __name__ == "__main__":
    from zeus.utils.coordinates import get_grid

    grid = get_grid(-90, 90, -180, 179.75)
    mask = region_mask_for_grid(grid, EUROPE_LATITUDE_RANGE, EUROPE_LONGITUDE_RANGE)

    lat_min, lat_max = EUROPE_LATITUDE_RANGE
    lon_min, lon_max = EUROPE_LONGITUDE_RANGE
    ones = mask == 1
    lats = grid[..., 0][ones]
    lons = grid[..., 1][ones]
    assert (lats >= lat_min).all() and (lats <= lat_max).all(), "lat out of range"
    assert (lons >= lon_min).all() and (lons <= lon_max).all(), "lon out of range"
    in_box = (
        (grid[..., 0] >= lat_min)
        & (grid[..., 0] <= lat_max)
        & (grid[..., 1] >= lon_min)
        & (grid[..., 1] <= lon_max)
    )
    assert (mask[in_box] == 1).all(), "inside box but not 1"
    print("mask.shape", mask.shape, "ones", mask.sum().item())
