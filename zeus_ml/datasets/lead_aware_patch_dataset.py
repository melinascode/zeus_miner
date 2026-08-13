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
            self._cycle_data.append(
                {
                    "metadata": metadata,
                    "inputs": inputs,
                    "residuals": residuals,
                    "lead_hours": lead_hours,
                    "lat_starts": lat_starts,
                    "lon_starts": lon_starts,
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
        else:
            gfs_mean, gfs_std, residual_std = self.statistics.tensors()
            model_input = (raw - gfs_mean) / gfs_std
            residual_target = residual / residual_std
        return {
            "cycle_key": ref.cycle_key,
            "lead_hour": torch.tensor(lead_hour, dtype=torch.float32),
            "context": context.to(torch.float32),
            "static_features": static_features,
            "metric_weights": metric_weights.unsqueeze(0),
            "raw": raw,
            "truth": raw + residual,
            "model_input": model_input,
            "residual_target": residual_target,
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
