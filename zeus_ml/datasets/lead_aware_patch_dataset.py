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
from zeus_ml.models.lead_aware_residual_cnn import (
    VARIABLES,
    build_static_features,
    build_temporal_context,
)


@dataclass(frozen=True)
class ChannelStatistics:
    gfs_mean: tuple[float, ...]
    gfs_std: tuple[float, ...]
    residual_std: tuple[float, ...]

    def __post_init__(self) -> None:
        expected = len(VARIABLES)
        if not (
            len(self.gfs_mean)
            == len(self.gfs_std)
            == len(self.residual_std)
            == expected
        ):
            raise ValueError(f"Channel statistics require {expected} values.")
        if min(self.gfs_std) <= 0.0 or min(self.residual_std) <= 0.0:
            raise ValueError("Standard deviations must be positive.")

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "gfs_mean": list(self.gfs_mean),
            "gfs_std": list(self.gfs_std),
            "residual_std": list(self.residual_std),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ChannelStatistics":
        return cls(
            gfs_mean=tuple(float(value) for value in payload["gfs_mean"]),
            gfs_std=tuple(float(value) for value in payload["gfs_std"]),
            residual_std=tuple(
                float(value) for value in payload["residual_std"]
            ),
        )

    def tensors(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (len(VARIABLES), 1, 1)
        return (
            torch.tensor(self.gfs_mean, device=device).view(shape),
            torch.tensor(self.gfs_std, device=device).view(shape),
            torch.tensor(self.residual_std, device=device).view(shape),
        )


@dataclass(frozen=True)
class PatchSampleRef:
    cycle_key: str
    cycle_index: int
    lead_index: int
    patch_index: int


class LeadAwarePatchDataset(Dataset):
    """Memory-mapped patch dataset produced by the v2 dataset builder."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        cycles: Sequence[str],
        statistics: ChannelStatistics | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.cycles = tuple(cycles)
        if not self.cycles:
            raise ValueError("At least one cycle is required.")
        self.statistics = statistics
        self._cycle_data: list[dict] = []
        self.samples: list[PatchSampleRef] = []
        for cycle_index, cycle_key in enumerate(self.cycles):
            cycle_dir = self.root_dir / cycle_key
            metadata_path = cycle_dir / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Missing metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if tuple(metadata["variables"]) != VARIABLES:
                raise ValueError(
                    f"Variable order mismatch for {cycle_key}: "
                    f"{metadata['variables']}"
                )
            inputs = np.load(cycle_dir / "inputs.npy", mmap_mode="r")
            residuals = np.load(cycle_dir / "residuals.npy", mmap_mode="r")
            lead_hours = np.load(cycle_dir / "lead_hours.npy", mmap_mode="r")
            lat_starts = np.load(cycle_dir / "lat_starts.npy", mmap_mode="r")
            lon_starts = np.load(cycle_dir / "lon_starts.npy", mmap_mode="r")
            zonal_path = cycle_dir / "zonal_means.npy"
            zonal_means = (
                np.load(zonal_path, mmap_mode="r") if zonal_path.is_file() else None
            )
            if inputs.shape != residuals.shape:
                raise ValueError(f"Input/target shape mismatch for {cycle_key}.")
            if inputs.ndim != 5 or inputs.shape[2] != len(VARIABLES):
                raise ValueError(
                    "Patch arrays must have shape "
                    "(lead, patch, variable, latitude, longitude)."
                )
            if tuple(lead_hours.shape) != (inputs.shape[0],):
                raise ValueError(f"Invalid lead_hours shape for {cycle_key}.")
            if tuple(lat_starts.shape) != inputs.shape[:2]:
                raise ValueError(f"Invalid lat_starts shape for {cycle_key}.")
            if tuple(lon_starts.shape) != inputs.shape[:2]:
                raise ValueError(f"Invalid lon_starts shape for {cycle_key}.")
            if zonal_means is not None and (
                zonal_means.shape[0] != inputs.shape[0]
                or zonal_means.shape[1] != len(VARIABLES)
            ):
                raise ValueError(f"Invalid zonal_means shape for {cycle_key}.")
            self._cycle_data.append(
                {
                    "metadata": metadata,
                    "inputs": inputs,
                    "residuals": residuals,
                    "lead_hours": lead_hours,
                    "lat_starts": lat_starts,
                    "lon_starts": lon_starts,
                    "zonal_means": zonal_means,
                }
            )
            for lead_index in range(inputs.shape[0]):
                for patch_index in range(inputs.shape[1]):
                    self.samples.append(
                        PatchSampleRef(
                            cycle_key=cycle_key,
                            cycle_index=cycle_index,
                            lead_index=lead_index,
                            patch_index=patch_index,
                        )
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def sample_weights(
        self,
        *,
        long_lead_gain: float = 4.0,
        long_lead_power: float = 2.0,
    ) -> torch.Tensor:
        """Upweight long leads and high validator-weight patch centers."""

        weights = torch.empty(len(self.samples), dtype=torch.float64)
        for index, ref in enumerate(self.samples):
            data = self._cycle_data[ref.cycle_index]
            lead = float(data["lead_hours"][ref.lead_index])
            metadata = data["metadata"]
            height = int(metadata["full_shape"][0])
            width = int(metadata["full_shape"][1])
            patch = int(metadata.get("patch_size", data["inputs"].shape[-1]))
            lat_start = int(data["lat_starts"][ref.lead_index, ref.patch_index])
            lon_start = int(data["lon_starts"][ref.lead_index, ref.patch_index])
            lat_center = -90.0 + (lat_start + patch / 2.0) * (180.0 / (height - 1))
            lon_center = -180.0 + ((lon_start + patch / 2.0) % width) * (
                360.0 / width
            )
            cycle = datetime.strptime(ref.cycle_key, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
            geo = _center_geographic_weight(
                lat_center,
                lon_center,
                include_germany=(
                    ValidatorFaithfulScorer.region_regime(cycle)
                    == "europe_germany"
                ),
            )
            lat_weight = max(float(np.cos(np.deg2rad(lat_center))), 0.0)
            lead_weight = 1.0 + long_lead_gain * (lead / 360.0) ** long_lead_power
            weights[index] = lead_weight * (0.25 + lat_weight * geo)
        return weights

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        ref = self.samples[index]
        data = self._cycle_data[ref.cycle_index]
        metadata = data["metadata"]
        raw = torch.from_numpy(
            np.array(
                data["inputs"][ref.lead_index, ref.patch_index],
                dtype=np.float32,
                copy=True,
            )
        )
        residual = torch.from_numpy(
            np.array(
                data["residuals"][ref.lead_index, ref.patch_index],
                dtype=np.float32,
                copy=True,
            )
        )
        lead_hour = float(data["lead_hours"][ref.lead_index])
        cycle = datetime.strptime(ref.cycle_key, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
        context = build_temporal_context(
            torch.tensor(lead_hour),
            torch.tensor(float(cycle.hour)),
            torch.tensor(float(cycle.timetuple().tm_yday)),
        )
        lat_start = int(data["lat_starts"][ref.lead_index, ref.patch_index])
        lon_start = int(data["lon_starts"][ref.lead_index, ref.patch_index])
        height, width = raw.shape[-2:]
        stored_zonal = data["zonal_means"]
        if stored_zonal is None:
            zonal_physical = raw.mean(dim=-1)
        else:
            zonal_physical = torch.from_numpy(
                np.array(stored_zonal[ref.lead_index], dtype=np.float32, copy=True)
            )
        full_height = int(metadata["full_shape"][0])
        full_width = int(metadata["full_shape"][1])
        latitudes = torch.linspace(-90.0, 90.0, full_height)[
            lat_start : lat_start + height
        ]
        lon_indices = torch.remainder(
            torch.arange(lon_start, lon_start + width),
            full_width,
        )
        longitudes = -180.0 + lon_indices.to(torch.float32) * (
            360.0 / full_width
        )
        geographic = _geographic_patch(
            latitudes,
            longitudes,
            include_germany=(
                ValidatorFaithfulScorer.region_regime(cycle)
                == "europe_germany"
            ),
        )
        static_features = build_static_features(
            latitudes,
            longitudes,
            geographic_weights=geographic,
        )
        latitude_weights = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)
        metric_weights = latitude_weights[:, None] * geographic
        metric_weights = metric_weights / float(
            metadata["global_metric_weight_mean"]
        )

        if self.statistics is None:
            model_input = raw
            residual_target = residual
            zonal_mean = zonal_physical
        else:
            gfs_mean, gfs_std, residual_std = self.statistics.tensors()
            model_input = (raw - gfs_mean) / gfs_std
            residual_target = residual / residual_std
            zonal_mean = (zonal_physical - gfs_mean.squeeze(-1)) / gfs_std.squeeze(-1)
        return {
            "cycle_key": ref.cycle_key,
            "lead_hour": torch.tensor(lead_hour, dtype=torch.float32),
            "lat_start": torch.tensor(lat_start, dtype=torch.int64),
            "context": context.to(torch.float32),
            "static_features": static_features,
            "metric_weights": metric_weights.unsqueeze(0),
            "raw": raw,
            "truth": raw + residual,
            "model_input": model_input,
            "residual_target": residual_target,
            "zonal_mean": zonal_mean.to(torch.float32),
        }


def estimate_channel_statistics(
    dataset: LeadAwarePatchDataset,
    *,
    max_samples: int = 2048,
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


def _geographic_patch(
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    *,
    include_germany: bool,
) -> torch.Tensor:
    lat_grid = latitudes[:, None]
    lon_grid = longitudes[None, :]
    europe = (
        (lat_grid >= 34.0)
        & (lat_grid <= 72.0)
        & (lon_grid >= -25.0)
        & (lon_grid <= 45.0)
    )
    germany = (
        (lat_grid >= 47.0)
        & (lat_grid <= 56.0)
        & (lon_grid >= 6.0)
        & (lon_grid <= 15.0)
    )
    weights = torch.ones(
        (latitudes.numel(), longitudes.numel()),
        dtype=torch.float32,
    )
    weights = torch.where(europe, 1.5, weights)
    return torch.where(germany, 2.5, weights) if include_germany else weights


def _center_geographic_weight(
    latitude: float,
    longitude: float,
    *,
    include_germany: bool,
) -> float:
    europe = 34.0 <= latitude <= 72.0 and -25.0 <= longitude <= 45.0
    germany = 47.0 <= latitude <= 56.0 and 6.0 <= longitude <= 15.0
    if include_germany and germany:
        return 2.5
    if europe:
        return 1.5
    return 1.0


EUROPE_LAT = (34.0, 72.0)
EUROPE_LON = (-25.0, 45.0)
GERMANY_LAT = (47.0, 56.0)
GERMANY_LON = (6.0, 15.0)


def sample_patch_origins(
    rng: np.random.Generator,
    *,
    n_leads: int,
    patches_per_lead: int,
    patch_size: int,
    include_germany: bool,
    full_height: int = 721,
    full_width: int = 1440,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample patch origins with Europe/Germany guarantees and metric weights."""

    if patches_per_lead < 2:
        raise ValueError("Need at least two patches per lead for region coverage.")
    cell_weights = cell_sampling_weights(
        full_height=full_height,
        full_width=full_width,
        include_germany=include_germany,
    )
    lat_starts = np.empty((n_leads, patches_per_lead), dtype=np.int32)
    lon_starts = np.empty_like(lat_starts)
    for lead in range(n_leads):
        lat_starts[lead, 0], lon_starts[lead, 0] = sample_box_origin(
            rng,
            lat_bounds=EUROPE_LAT,
            lon_bounds=EUROPE_LON,
            patch_size=patch_size,
            full_height=full_height,
            full_width=full_width,
        )
        if include_germany:
            lat_starts[lead, 1], lon_starts[lead, 1] = sample_box_origin(
                rng,
                lat_bounds=GERMANY_LAT,
                lon_bounds=GERMANY_LON,
                patch_size=patch_size,
                full_height=full_height,
                full_width=full_width,
            )
            remaining_start = 2
        else:
            remaining_start = 1
        remaining = patches_per_lead - remaining_start
        if remaining > 0:
            chosen = rng.choice(
                cell_weights.size,
                size=remaining,
                replace=True,
                p=cell_weights.ravel(),
            )
            lats, lons = np.divmod(chosen, full_width)
            lat_starts[lead, remaining_start:] = np.clip(
                lats - patch_size // 2,
                0,
                full_height - patch_size,
            )
            lon_starts[lead, remaining_start:] = np.clip(
                lons - patch_size // 2,
                0,
                full_width - patch_size,
            )
    return lat_starts, lon_starts


def sample_box_origin(
    rng: np.random.Generator,
    *,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    patch_size: int,
    full_height: int,
    full_width: int,
) -> tuple[int, int]:
    lat = float(rng.uniform(*lat_bounds))
    lon = float(rng.uniform(*lon_bounds))
    lat_index = int(round((lat + 90.0) / 0.25))
    lon_index = int(round((lon + 180.0) / 0.25)) % full_width
    lat_start = int(np.clip(lat_index - patch_size // 2, 0, full_height - patch_size))
    lon_start = int(np.clip(lon_index - patch_size // 2, 0, full_width - patch_size))
    return lat_start, lon_start


def cell_sampling_weights(
    *,
    full_height: int,
    full_width: int,
    include_germany: bool,
) -> np.ndarray:
    latitudes = np.linspace(-90.0, 90.0, full_height)
    longitudes = np.arange(-180.0, 180.0, 0.25)[:full_width]
    latitude = np.clip(np.cos(np.deg2rad(latitudes)), 0.0, None)[:, None]
    lat_grid = latitudes[:, None]
    lon_grid = longitudes[None, :]
    europe = (
        (lat_grid >= EUROPE_LAT[0])
        & (lat_grid <= EUROPE_LAT[1])
        & (lon_grid >= EUROPE_LON[0])
        & (lon_grid <= EUROPE_LON[1])
    )
    germany = (
        (lat_grid >= GERMANY_LAT[0])
        & (lat_grid <= GERMANY_LAT[1])
        & (lon_grid >= GERMANY_LON[0])
        & (lon_grid <= GERMANY_LON[1])
    )
    geographic = np.ones((full_height, full_width), dtype=np.float64)
    geographic[europe] = 1.5
    if include_germany:
        geographic[germany] = 2.5
    weights = latitude * geographic
    return weights / weights.sum()
