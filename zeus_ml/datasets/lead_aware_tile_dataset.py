from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from evaluation.scoring import ValidatorFaithfulScorer
from zeus.utils.coordinates import get_grid
from zeus.utils.region_mask import (
    OLD_REGION_CONFIGS,
    REGION_CONFIGS,
    build_geographic_weights,
)
from zeus_ml.datasets.lead_aware_patch_dataset import ChannelStatistics
from zeus_ml.models.lead_aware_residual_cnn import VARIABLES
from zeus_ml.models.lead_aware_residual_cnn_v4 import (
    build_v4_static_features,
    cosine_solar_zenith,
    crop_native_tile,
    downsample_2deg,
    load_static_maps,
    valid_time_from_cycle,
)


TILE_SIZE = 512
FULL_HEIGHT = 721
FULL_WIDTH = 1440
LAT_STARTS = (0, 209)
LON_STARTS = (0, 480, 960)


@dataclass(frozen=True)
class TileSampleRef:
    cycle_key: str
    cycle_index: int
    lead_index: int
    lat_start: int
    lon_start: int


class LeadAwareTileDataset(Dataset):
    """Full-globe float16 cycle cache cropped to 512 tiles plus a 2° Earth."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        cycles: Sequence[str],
        static_root: str | Path,
        statistics: ChannelStatistics | None = None,
        tile_size: int = TILE_SIZE,
        lat_starts: Sequence[int] = LAT_STARTS,
        lon_starts: Sequence[int] = LON_STARTS,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.cycles = tuple(cycles)
        self.statistics = statistics
        self.tile_size = tile_size
        self.lat_starts = tuple(int(value) for value in lat_starts)
        self.lon_starts = tuple(int(value) for value in lon_starts)
        self.land, self.orography = load_static_maps(static_root)
        self.latitudes = torch.linspace(-90.0, 90.0, FULL_HEIGHT)
        self.longitudes = torch.arange(-180.0, 180.0, 0.25)
        self._cycle_data: list[dict] = []
        self.samples: list[TileSampleRef] = []
        for cycle_index, cycle_key in enumerate(self.cycles):
            cycle_dir = self.root_dir / cycle_key
            metadata = json.loads(
                (cycle_dir / "metadata.json").read_text(encoding="utf-8")
            )
            if tuple(metadata["variables"]) != VARIABLES:
                raise ValueError(f"Unexpected variables in {cycle_key}.")
            inputs = np.load(cycle_dir / "inputs.npy", mmap_mode="r")
            residuals = np.load(cycle_dir / "residuals.npy", mmap_mode="r")
            zonal = np.load(cycle_dir / "zonal_means.npy", mmap_mode="r")
            n_leads = int(inputs.shape[0])
            cycle_time = datetime.strptime(
                cycle_key, "%Y%m%dT%H%M%SZ"
            ).replace(tzinfo=timezone.utc)
            grid = get_grid(-90.0, 90.0, -180.0, 179.75)
            regime = ValidatorFaithfulScorer.region_regime(cycle_time)
            configs = (
                OLD_REGION_CONFIGS if regime == "europe_only" else REGION_CONFIGS
            )
            geographic = build_geographic_weights(grid, configs)
            latitude_weight = torch.cos(
                torch.deg2rad(self.latitudes)
            ).clamp_min(0.0)
            global_metric_mean = float(
                (latitude_weight[:, None] * geographic).mean()
            )
            self._cycle_data.append(
                {
                    "cycle_key": cycle_key,
                    "cycle_time": cycle_time,
                    "inputs": inputs,
                    "residuals": residuals,
                    "zonal": zonal,
                    "geographic": geographic,
                    "global_metric_mean": global_metric_mean,
                    "include_germany": regime == "europe_germany",
                    "n_leads": n_leads,
                }
            )
            for lead in range(n_leads):
                for lat_start in self.lat_starts:
                    for lon_start in self.lon_starts:
                        self.samples.append(
                            TileSampleRef(
                                cycle_key=cycle_key,
                                cycle_index=cycle_index,
                                lead_index=lead,
                                lat_start=lat_start,
                                lon_start=lon_start,
                            )
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def sample_weights(self) -> torch.Tensor:
        weights = []
        for sample in self.samples:
            lead = sample.lead_index
            lead_weight = 1.0 + 4.0 * (lead / 360.0) ** 2
            lat0 = sample.lat_start + self.tile_size // 2
            lon0 = (sample.lon_start + self.tile_size // 2) % FULL_WIDTH
            geo = float(
                self._cycle_data[sample.cycle_index]["geographic"][lat0, lon0]
            )
            weights.append(lead_weight * geo)
        return torch.tensor(weights, dtype=torch.double)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[index]
        data = self._cycle_data[sample.cycle_index]
        raw = torch.from_numpy(
            np.asarray(data["inputs"][sample.lead_index], dtype=np.float32)
        )
        residual = torch.from_numpy(
            np.asarray(data["residuals"][sample.lead_index], dtype=np.float32)
        )
        zonal = torch.from_numpy(
            np.asarray(data["zonal"][sample.lead_index], dtype=np.float32)
        )
        truth = raw + residual
        lat_start = sample.lat_start
        lon_start = sample.lon_start
        raw_tile = crop_native_tile(
            raw,
            lat_start=lat_start,
            lon_start=lon_start,
            height=self.tile_size,
            width=self.tile_size,
        )
        truth_tile = crop_native_tile(
            truth,
            lat_start=lat_start,
            lon_start=lon_start,
            height=self.tile_size,
            width=self.tile_size,
        )
        geographic = data["geographic"]
        geo_tile = crop_native_tile(
            geographic,
            lat_start=lat_start,
            lon_start=lon_start,
            height=self.tile_size,
            width=self.tile_size,
        )
        land_tile = crop_native_tile(
            self.land,
            lat_start=lat_start,
            lon_start=lon_start,
            height=self.tile_size,
            width=self.tile_size,
        )
        oro_tile = crop_native_tile(
            self.orography,
            lat_start=lat_start,
            lon_start=lon_start,
            height=self.tile_size,
            width=self.tile_size,
        )
        lead_hour = float(sample.lead_index)
        valid_time = valid_time_from_cycle(data["cycle_time"], lead_hour)
        zenith = cosine_solar_zenith(
            self.latitudes,
            self.longitudes,
            valid_time,
        )
        zenith_tile = crop_native_tile(
            zenith,
            lat_start=lat_start,
            lon_start=lon_start,
            height=self.tile_size,
            width=self.tile_size,
        )
        latitudes = self.latitudes[lat_start : lat_start + self.tile_size]
        longitudes = -180.0 + (
            (lon_start + torch.arange(self.tile_size)) % FULL_WIDTH
        ).to(torch.float32) * 0.25
        static_tile = build_v4_static_features(
            latitudes,
            longitudes,
            geographic_weights=geo_tile,
            land_sea=land_tile,
            orography=oro_tile,
            zenith=zenith_tile,
        )
        metric_weights = (
            torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)[:, None]
            * geo_tile
        )
        metric_weights = metric_weights / data["global_metric_mean"]
        coarse_weather = downsample_2deg(raw)
        coarse_static = downsample_2deg(
            build_v4_static_features(
                self.latitudes,
                self.longitudes,
                geographic_weights=geographic,
                land_sea=self.land,
                orography=self.orography,
                zenith=zenith,
            )
        )
        if self.statistics is None:
            model_input = raw_tile
            zonal_mean = zonal
        else:
            gfs_mean, gfs_std, residual_std = self.statistics.tensors()
            model_input = (raw_tile - gfs_mean) / gfs_std
            zonal_mean = (zonal - gfs_mean.squeeze(-1)) / gfs_std.squeeze(-1)
            coarse_weather = (coarse_weather - gfs_mean.squeeze(-1).unsqueeze(-1)) / (
                gfs_std.squeeze(-1).unsqueeze(-1)
            )
        from zeus_ml.models.lead_aware_residual_cnn import build_temporal_context

        context = build_temporal_context(
            torch.tensor(lead_hour),
            torch.tensor(float(data["cycle_time"].hour)),
            torch.tensor(float(data["cycle_time"].timetuple().tm_yday)),
        )
        return {
            "cycle_key": sample.cycle_key,
            "lead_hour": torch.tensor(lead_hour, dtype=torch.float32),
            "lat_start": torch.tensor(lat_start, dtype=torch.int64),
            "lon_start": torch.tensor(lon_start, dtype=torch.int64),
            "context": context.to(torch.float32),
            "static_features": static_tile,
            "metric_weights": metric_weights.unsqueeze(0),
            "raw": raw_tile,
            "truth": truth_tile,
            "model_input": model_input,
            "zonal_mean": zonal_mean.to(torch.float32),
            "coarse_input": torch.cat((coarse_weather, coarse_static), dim=0),
        }


def estimate_tile_statistics(
    dataset: LeadAwareTileDataset,
    *,
    max_samples: int = 512,
    seed: int = 17,
) -> ChannelStatistics:
    if len(dataset) < 1:
        raise ValueError("Cannot estimate statistics from an empty dataset.")
    count = min(max_samples, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count]
    sums = torch.zeros(len(VARIABLES), dtype=torch.float64)
    squares = torch.zeros_like(sums)
    residual_squares = torch.zeros_like(sums)
    cells = 0
    for index in indices.tolist():
        sample = dataset[index]
        raw = sample["raw"].to(torch.float64)
        residual = (sample["truth"] - sample["raw"]).to(torch.float64)
        sums += raw.sum(dim=(1, 2))
        squares += raw.square().sum(dim=(1, 2))
        residual_squares += residual.square().sum(dim=(1, 2))
        cells += raw.shape[-2] * raw.shape[-1]
    mean = sums / cells
    variance = (squares / cells - mean.square()).clamp_min(1e-12)
    residual_scale = (residual_squares / cells).clamp_min(1e-12).sqrt()
    return ChannelStatistics(
        gfs_mean=tuple(mean.tolist()),
        gfs_std=tuple(variance.sqrt().tolist()),
        residual_std=tuple(residual_scale.tolist()),
    )
