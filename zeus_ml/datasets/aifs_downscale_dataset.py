"""Training samples that pair interpolated AIFS forecasts with ERA5 truth.

A sample is one (cycle, valid hour) drawn from a 00z AIFS Single run, cropped
to several random tiles. AIFS GRIB files decode in a few seconds, so cycles are
decoded on demand into a small in-memory cache rather than duplicated to disk;
the sampler visits many leads per cycle so that cost amortizes away.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler

from zeus.utils.coordinates import get_grid
from zeus.utils.region_mask import (
    REGION_CONFIGS,
    build_geographic_weights,
    geographic_scalar_for_variable,
)
from zeus.validator.constants import (
    EUROPE_LATITUDE_RANGE,
    EUROPE_LONGITUDE_RANGE,
)
from zeus_ml.models.aifs_downscaler_cnn import (
    FULL_HEIGHT,
    FULL_WIDTH,
    MAX_LEAD_HOURS,
    STEP_HOURS,
    VARIABLES,
    DownscalerStatistics,
    bracket_for_lead,
    build_downscaler_context,
    build_downscaler_static_features,
    load_static_maps,
    zenith_triplet,
)
from zeus_ml.models.lead_aware_residual_cnn_v4 import crop_native_tile


GRIB_SHORT_NAMES = ("2t", "100u", "100v")
ERA5_SHORT_CODES = ("t2m", "u100", "v100")
N_STEPS = MAX_LEAD_HOURS // STEP_HOURS + 1


def to_zeus_grid(field: np.ndarray) -> np.ndarray:
    """ECMWF open data GRIB -> Zeus grid. Already starts at -180 longitude."""
    return np.ascontiguousarray(field[::-1, :])


def era5_to_zeus_grid(field: np.ndarray) -> np.ndarray:
    """ERA5 NetCDF (lat 90..-90, lon 0..360) -> Zeus grid."""
    return np.ascontiguousarray(np.roll(field[::-1, :], FULL_WIDTH // 2, axis=1))


class AifsCycleReader:
    """Decode AIFS GRIB runs to fp16 arrays, keeping a few cycles resident."""

    def __init__(self, root: str | Path, *, cache_size: int = 2) -> None:
        self.root = Path(root)
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def path_for(self, cycle_key: str) -> Path:
        return self.root / f"{cycle_key}.grib2"

    def available_cycles(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.grib2"))

    def get(self, cycle_key: str) -> np.ndarray:
        """Return an fp16 array of shape (61, 3, 721, 1440) on the Zeus grid."""

        cached = self._cache.get(cycle_key)
        if cached is not None:
            self._cache.move_to_end(cycle_key)
            return cached
        array = self._decode(self.path_for(cycle_key))
        self._cache[cycle_key] = array
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return array

    @staticmethod
    def _decode(path: Path) -> np.ndarray:
        import eccodes

        out = np.zeros((N_STEPS, len(VARIABLES), FULL_HEIGHT, FULL_WIDTH), np.float16)
        seen = np.zeros((N_STEPS, len(VARIABLES)), bool)
        index = {name: i for i, name in enumerate(GRIB_SHORT_NAMES)}
        with open(path, "rb") as handle:
            while True:
                gid = eccodes.codes_grib_new_from_file(handle)
                if gid is None:
                    break
                short = eccodes.codes_get(gid, "shortName")
                step = int(eccodes.codes_get(gid, "endStep"))
                if short in index and step % STEP_HOURS == 0:
                    s = step // STEP_HOURS
                    v = index[short]
                    values = eccodes.codes_get_values(gid).reshape(
                        FULL_HEIGHT, FULL_WIDTH
                    )
                    out[s, v] = to_zeus_grid(values).astype(np.float16)
                    seen[s, v] = True
                eccodes.codes_release(gid)
        if not seen.all():
            missing = np.argwhere(~seen)
            raise ValueError(f"{path.name} is missing {len(missing)} fields.")
        return out


class EnsMeanCycleReader:
    """Read precomputed AIFS-ENS mean cycles stored as fp16 ``.npy`` arrays."""

    def __init__(self, root: str | Path, *, cache_size: int = 2) -> None:
        self.root = Path(root)
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def path_for(self, cycle_key: str) -> Path:
        return self.root / f"{cycle_key}.npy"

    def available_cycles(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.npy"))

    def get(self, cycle_key: str) -> np.ndarray:
        """Return an fp16 array of shape (61, 3, 721, 1440) on the Zeus grid."""

        cached = self._cache.get(cycle_key)
        if cached is not None:
            self._cache.move_to_end(cycle_key)
            return cached
        array = np.load(self.path_for(cycle_key))
        expected = (N_STEPS, len(VARIABLES), FULL_HEIGHT, FULL_WIDTH)
        if array.shape != expected:
            raise ValueError(
                f"{cycle_key}.npy has shape {array.shape}, expected {expected}."
            )
        self._cache[cycle_key] = array
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return array


class Era5HourlyReader:
    """Read single ERA5 hours from the daily NetCDF archive."""

    def __init__(self, root: str | Path, *, cache_size: int = 12) -> None:
        self.root = Path(root)
        self.cache_size = max(1, int(cache_size))
        self._open: OrderedDict[tuple[str, str], xr.Dataset] = OrderedDict()

    def _dataset(self, variable: str, day: str) -> xr.Dataset:
        key = (variable, day)
        cached = self._open.get(key)
        if cached is not None:
            self._open.move_to_end(key)
            return cached
        path = self.root / variable / f"era5_{day}.nc"
        dataset = xr.open_dataset(path, engine="h5netcdf")
        self._open[key] = dataset
        while len(self._open) > self.cache_size:
            _, stale = self._open.popitem(last=False)
            stale.close()
        return dataset

    def has(self, valid_time: datetime) -> bool:
        day = valid_time.strftime("%Y-%m-%d")
        return all(
            (self.root / variable / f"era5_{day}.nc").is_file()
            for variable in VARIABLES
        )

    def read(self, valid_time: datetime) -> np.ndarray:
        """Return truth of shape (3, 721, 1440) in validator units."""

        day = valid_time.strftime("%Y-%m-%d")
        out = np.empty((len(VARIABLES), FULL_HEIGHT, FULL_WIDTH), np.float32)
        for i, (variable, code) in enumerate(zip(VARIABLES, ERA5_SHORT_CODES)):
            dataset = self._dataset(variable, day)
            values = dataset[code].isel(valid_time=valid_time.hour).values
            out[i] = era5_to_zeus_grid(np.asarray(values, dtype=np.float32))
        return out


@dataclass(frozen=True)
class PlanEntry:
    cycle_index: int
    lead_hour: int


class AifsDownscaleDataset(Dataset):
    """One item is a (cycle, lead) pair rendered as a group of random tiles."""

    def __init__(
        self,
        *,
        aifs_root: str | Path,
        era5_root: str | Path,
        static_root: str | Path,
        cycles: Sequence[str],
        tile_size: int = 192,
        tiles_per_item: int = 6,
        europe_fraction: float = 0.35,
        statistics: DownscalerStatistics | None = None,
        seed: int = 0,
        grib_cache_size: int = 2,
        use_lagged: bool = False,
        ens_root: str | Path | None = None,
        geo_mode: str = "boxes",
    ) -> None:
        # With ens_root the primary forecast is the ENS mean and the paired
        # ("lagged") channels carry the same-day AIFS Single run instead of
        # yesterday's run: ENS cycles are weekly, so no previous-day ENS
        # exists, and the single run is what serving has next to the mean.
        self.ens_mode = ens_root is not None
        self.single_reader = AifsCycleReader(aifs_root, cache_size=grib_cache_size)
        self.reader = (
            EnsMeanCycleReader(ens_root, cache_size=grib_cache_size)
            if self.ens_mode
            else self.single_reader
        )
        self.truth = Era5HourlyReader(era5_root)
        self.cycles = tuple(cycles)
        self.use_lagged = bool(use_lagged)
        self.tile_size = int(tile_size)
        self.tiles_per_item = int(tiles_per_item)
        self.europe_fraction = float(europe_fraction)
        self.statistics = statistics
        self.rng = np.random.default_rng(seed)

        land, orography, roughness = load_static_maps(static_root)
        self.land = land
        self.orography = orography
        self.roughness = roughness
        self.latitudes = torch.linspace(-90.0, 90.0, FULL_HEIGHT)
        self.longitudes = torch.arange(-180.0, 180.0, 0.25)

        # Score under the region rules in force today, not the rules that
        # applied when each historical cycle was issued.
        grid = get_grid(-90.0, 90.0, -180.0, 179.75)
        self.geographic = build_geographic_weights(grid, REGION_CONFIGS)
        latitude_weight = torch.cos(torch.deg2rad(self.latitudes)).clamp_min(0.0)
        self.global_metric_mean = float(
            (latitude_weight[:, None] * self.geographic).mean()
        )
        if geo_mode == "official":
            # Post-2026-08-25 validator metric: per-variable capacity scalars,
            # each normalized by its own global mean like custom_rmse does on
            # full-globe challenges. Channels follow VARIABLES (t2m, u100, v100).
            channels = []
            for variable in ("2m_temperature", "100m_u_component_of_wind",
                             "100m_v_component_of_wind"):
                metric = latitude_weight[:, None] * geographic_scalar_for_variable(
                    variable
                )
                channels.append(metric / metric.mean())
            self.metric_map = torch.stack(channels)
        elif geo_mode == "boxes":
            self.metric_map = (
                latitude_weight[:, None] * self.geographic / self.global_metric_mean
            ).unsqueeze(0)
        else:
            raise ValueError(f"Unknown geo_mode {geo_mode!r}")
        self.cycle_times = [
            datetime.strptime(key, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            for key in self.cycles
        ]
        # Previous-day run for the lagged-pair input. Buffer-zone runs are fine
        # here: the lagged forecast targets the same valid time, so no extra
        # ERA5 truth enters the sample.
        self._prev_key: dict[str, str | None] = {}
        for key, cycle_time in zip(self.cycles, self.cycle_times):
            if self.ens_mode:
                pair = key
            else:
                pair = (cycle_time - timedelta(days=1)).strftime("%Y%m%dT%H%M%SZ")
            self._prev_key[key] = (
                pair if self.single_reader.path_for(pair).is_file() else None
            )
        self.entries: list[PlanEntry] = [
            PlanEntry(cycle_index=c, lead_hour=lead)
            for c in range(len(self.cycles))
            for lead in range(MAX_LEAD_HOURS + 1)
        ]
        self._europe_bounds = self._europe_tile_bounds()

    def __len__(self) -> int:
        return len(self.entries)

    def _europe_tile_bounds(self) -> tuple[int, int, int, int]:
        lat_lo, lat_hi = EUROPE_LATITUDE_RANGE
        lon_lo, lon_hi = EUROPE_LONGITUDE_RANGE
        half = self.tile_size // 2
        lat_start_lo = int((lat_lo + 90.0) / 0.25) - half
        lat_start_hi = int((lat_hi + 90.0) / 0.25) - half
        lon_start_lo = int((lon_lo + 180.0) / 0.25) - half
        lon_start_hi = int((lon_hi + 180.0) / 0.25) - half
        max_lat_start = FULL_HEIGHT - self.tile_size
        return (
            max(0, min(lat_start_lo, max_lat_start)),
            max(0, min(lat_start_hi, max_lat_start)),
            lon_start_lo,
            lon_start_hi,
        )

    def _sample_origins(self) -> list[tuple[int, int]]:
        max_lat_start = FULL_HEIGHT - self.tile_size
        e_lat_lo, e_lat_hi, e_lon_lo, e_lon_hi = self._europe_bounds
        origins = []
        for _ in range(self.tiles_per_item):
            if self.rng.random() < self.europe_fraction:
                lat_start = int(self.rng.integers(e_lat_lo, e_lat_hi + 1))
                lon_start = int(self.rng.integers(e_lon_lo, e_lon_hi + 1))
            else:
                lat_start = int(self.rng.integers(0, max_lat_start + 1))
                lon_start = int(self.rng.integers(0, FULL_WIDTH))
            origins.append((lat_start, lon_start % FULL_WIDTH))
        return origins

    def _build_fields(self, cycle_index: int, lead: int) -> dict[str, torch.Tensor]:
        """Interpolated state, bracket tendency, truth, zenith and context."""

        cycle_key = self.cycles[cycle_index]
        cycle_time = self.cycle_times[cycle_index]
        left, right, fraction = bracket_for_lead(lead)
        cycle = self.reader.get(cycle_key)
        a_left = torch.from_numpy(cycle[left].astype(np.float32))
        a_right = torch.from_numpy(cycle[right].astype(np.float32))
        interpolated = (1.0 - fraction) * a_left + fraction * a_right
        lagged = None
        if self.use_lagged:
            prev_key = self._prev_key[cycle_key]
            lag_lead = lead if self.ens_mode else lead + 24
            if prev_key is not None and lag_lead <= MAX_LEAD_HOURS:
                prev = self.single_reader.get(prev_key)
                l2, r2, f2 = bracket_for_lead(lag_lead)
                lagged = (1.0 - f2) * torch.from_numpy(
                    prev[l2].astype(np.float32)
                ) + f2 * torch.from_numpy(prev[r2].astype(np.float32))
            else:
                # No usable paired forecast: duplicate the current state so the
                # difference channel is exactly zero ("no extra information").
                lagged = interpolated
        zenith, zenith_anomaly = zenith_triplet(
            self.latitudes, self.longitudes, cycle_time, lead
        )
        return {
            "cycle_key": cycle_key,
            "interpolated": interpolated,
            "lagged": lagged,
            "delta": a_right - a_left,
            "truth": torch.from_numpy(
                self.truth.read(cycle_time + timedelta(hours=lead))
            ),
            "zenith": zenith,
            "zenith_anomaly": zenith_anomaly,
            "context": build_downscaler_context(
                lead_hour=lead,
                cycle_hour=cycle_time.hour,
                day_of_year=cycle_time.timetuple().tm_yday,
                fraction=fraction,
            ),
        }

    def _normalize_input(
        self,
        raw: torch.Tensor,
        delta: torch.Tensor,
        lagged: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.statistics is None:
            blocks = [raw, delta]
            if lagged is not None:
                blocks += [lagged, lagged - raw]
            return torch.cat(blocks, dim=0)
        mean, std, delta_std, _ = self.statistics.tensors()
        blocks = [(raw - mean) / std, delta / delta_std]
        if lagged is not None:
            blocks += [(lagged - mean) / std, (lagged - raw) / delta_std]
        return torch.cat(blocks, dim=0)

    def full_globe_item(self, cycle_index: int, lead: int) -> dict[str, torch.Tensor]:
        """One full 721x1440 sample, matching how the model is actually served."""

        fields = self._build_fields(cycle_index, lead)
        static = build_downscaler_static_features(
            self.latitudes,
            self.longitudes,
            geographic_weights=self.geographic,
            land_sea=self.land,
            orography=self.orography,
            roughness=self.roughness,
            zenith=fields["zenith"],
            zenith_anomaly=fields["zenith_anomaly"],
        )
        return {
            "cycle_key": fields["cycle_key"],
            "lead_hour": torch.tensor([float(lead)]),
            "context": fields["context"].unsqueeze(0),
            "model_input": self._normalize_input(
                fields["interpolated"], fields["delta"], fields["lagged"]
            ).unsqueeze(0),
            "raw": fields["interpolated"].unsqueeze(0),
            "truth": fields["truth"].unsqueeze(0),
            "static_features": static.unsqueeze(0),
            "metric_weights": self.metric_map.unsqueeze(0),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.entries[index]
        lead = entry.lead_hour
        fields = self._build_fields(entry.cycle_index, lead)
        interpolated = fields["interpolated"]
        lagged = fields["lagged"]
        delta = fields["delta"]
        truth = fields["truth"]
        zenith = fields["zenith"]
        zenith_anomaly = fields["zenith_anomaly"]
        context = fields["context"]
        cycle_key = fields["cycle_key"]

        origins = self._sample_origins()
        raw_tiles, truth_tiles, input_tiles, static_tiles, weight_tiles = (
            [],
            [],
            [],
            [],
            [],
        )
        size = self.tile_size
        for lat_start, lon_start in origins:
            crop = lambda field: crop_native_tile(  # noqa: E731
                field, lat_start=lat_start, lon_start=lon_start, height=size, width=size
            )
            raw_tile = crop(interpolated)
            delta_tile = crop(delta)
            truth_tile = crop(truth)
            geo_tile = crop(self.geographic)
            latitudes = self.latitudes[lat_start : lat_start + size]
            longitudes = (
                -180.0
                + ((lon_start + torch.arange(size)) % FULL_WIDTH).to(torch.float32)
                * 0.25
            )
            static_tile = build_downscaler_static_features(
                latitudes,
                longitudes,
                geographic_weights=geo_tile,
                land_sea=crop(self.land),
                orography=crop(self.orography),
                roughness=crop(self.roughness),
                zenith=crop(zenith),
                zenith_anomaly=crop(zenith_anomaly),
            )
            metric = crop(self.metric_map)
            model_input = self._normalize_input(
                raw_tile,
                delta_tile,
                crop(lagged) if lagged is not None else None,
            )
            raw_tiles.append(raw_tile)
            truth_tiles.append(truth_tile)
            input_tiles.append(model_input)
            static_tiles.append(static_tile)
            weight_tiles.append(metric)

        group = len(origins)
        return {
            "cycle_key": cycle_key,
            "lead_hour": torch.full((group,), float(lead)),
            "context": context.unsqueeze(0).expand(group, -1).contiguous(),
            "model_input": torch.stack(input_tiles),
            "raw": torch.stack(raw_tiles),
            "truth": torch.stack(truth_tiles),
            "static_features": torch.stack(static_tiles),
            "metric_weights": torch.stack(weight_tiles),
        }


class CycleBlockSampler(Sampler[int]):
    """Visit a few cycles per epoch, taking many leads from each.

    Random access across cycles would re-decode a GRIB file on every step, so
    leads are grouped by cycle. Leads are drawn without replacement inside a
    cycle to keep the diurnal phase varied.
    """

    def __init__(
        self,
        dataset: AifsDownscaleDataset,
        *,
        cycles_per_epoch: int,
        leads_per_cycle: int,
        seed: int = 0,
        midhour_boost: float = 0.0,
    ) -> None:
        self.n_cycles = len(dataset.cycles)
        self.n_leads = MAX_LEAD_HOURS + 1
        self.cycles_per_epoch = min(int(cycles_per_epoch), self.n_cycles)
        self.leads_per_cycle = min(int(leads_per_cycle), self.n_leads)
        self.seed = int(seed)
        self.epoch = 0
        # Production pays 15-19% extra MAE at hours off the 6h model steps;
        # midhour_boost > 0 oversamples those leads during training.
        weights = np.ones(self.n_leads, dtype=np.float64)
        weights[np.arange(self.n_leads) % 6 != 0] += float(midhour_boost)
        self.lead_probs = weights / weights.sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.cycles_per_epoch * self.leads_per_cycle

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + 1000 * self.epoch)
        cycles = rng.permutation(self.n_cycles)[: self.cycles_per_epoch]
        for cycle in cycles:
            leads = rng.choice(
                self.n_leads,
                size=self.leads_per_cycle,
                replace=False,
                p=self.lead_probs,
            )
            for lead in leads:
                yield int(cycle) * self.n_leads + int(lead)


def estimate_statistics(
    dataset: AifsDownscaleDataset,
    *,
    n_items: int = 48,
    seed: int = 17,
) -> DownscalerStatistics:
    """Estimate input and residual scales from raw (unnormalized) samples."""

    if dataset.statistics is not None:
        raise ValueError("Estimate statistics from a dataset without statistics.")
    rng = np.random.default_rng(seed)
    n_vars = len(VARIABLES)
    total = np.zeros(n_vars)
    squares = np.zeros(n_vars)
    delta_squares = np.zeros(n_vars)
    residual_squares = np.zeros(n_vars)
    cells = 0
    cycles = rng.permutation(len(dataset.cycles))[:n_items]
    for cycle in cycles:
        lead = int(rng.integers(0, MAX_LEAD_HOURS + 1))
        item = dataset[int(cycle) * (MAX_LEAD_HOURS + 1) + lead]
        raw = item["raw"].numpy().astype(np.float64)
        delta = item["model_input"].numpy()[:, n_vars : 2 * n_vars].astype(np.float64)
        residual = (item["truth"] - item["raw"]).numpy().astype(np.float64)
        total += raw.sum(axis=(0, 2, 3))
        squares += np.square(raw).sum(axis=(0, 2, 3))
        delta_squares += np.square(delta).sum(axis=(0, 2, 3))
        residual_squares += np.square(residual).sum(axis=(0, 2, 3))
        cells += raw.shape[0] * raw.shape[2] * raw.shape[3]
    mean = total / cells
    variance = np.maximum(squares / cells - np.square(mean), 1e-12)
    return DownscalerStatistics(
        state_mean=tuple(mean.tolist()),
        state_std=tuple(np.sqrt(variance).tolist()),
        delta_std=tuple(np.sqrt(np.maximum(delta_squares / cells, 1e-12)).tolist()),
        residual_std=tuple(
            np.sqrt(np.maximum(residual_squares / cells, 1e-12)).tolist()
        ),
    )
