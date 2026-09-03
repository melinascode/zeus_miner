"""Evaluate v2_ens (live) and v3_wind on the identical held-out validation set.

Reproduces the exact validation split and entries of the V3 training run
(same cycle list, ERA5-coverage filter, seeds) and scores both checkpoints
with the official geo scalars, so the numbers are directly comparable.

Run on the GPU pod:
  PYTHONPATH=/workspace/zeus /workspace/venv/bin/python tools/compare_v2_v3_val.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from zeus_ml.datasets.aifs_downscale_dataset import (
    AifsDownscaleDataset,
    Era5HourlyReader,
)
from zeus_ml.models.aifs_downscaler_cnn import (
    MAX_LEAD_HOURS,
    AifsDownscalerCNN,
    DownscalerStatistics,
)
from zeus_ml.train.train_aifs_downscaler import (
    SHORT_NAMES,
    build_split,
    evaluate,
    validation_entries,
)

ROOT = Path("/workspace/data/evaluation")
CHECKPOINTS = {
    "v2_ens (live)": ROOT / "training/aifs_downscaler_v2_ens.pt",
    "v3_wind (new)": ROOT / "training/aifs_downscaler_v3_wind.pt",
}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Same cycle selection as the training run.
    all_cycles = sorted(p.stem for p in (ROOT / "aifs_ens_mean").glob("*.npy"))
    era5 = Era5HourlyReader(ROOT / "era5")
    all_cycles = [
        key
        for key in all_cycles
        if all(
            era5.has(
                datetime.strptime(key, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                + timedelta(hours=hour)
            )
            for hour in range(0, MAX_LEAD_HOURS + 1, 24)
        )
    ]
    split = build_split(all_cycles)
    statistics = DownscalerStatistics.from_dict(
        json.loads(
            (ROOT / "training/aifs_downscaler_v2_ens.statistics.json").read_text()
        )
    )
    # seed=0+1 and entry seed defaults match main()'s validation setup.
    dataset = AifsDownscaleDataset(
        aifs_root=ROOT / "aifs_single",
        ens_root=ROOT / "aifs_ens_mean",
        era5_root=ROOT / "era5",
        static_root=ROOT / "training/aifs_static",
        cycles=split.validation,
        europe_fraction=0.50,
        statistics=statistics,
        seed=1,
        grib_cache_size=3,
        use_lagged=True,
        geo_mode="official",
    )
    entries = validation_entries(min(12, len(dataset.cycles)), 6)
    residual_scales = (
        torch.tensor(statistics.residual_std, dtype=torch.float32)
        .view(-1, 1, 1)
        .to(device)
    )
    print(f"validation: {len(dataset.cycles)} cycles, {len(entries)} samples")
    for name, path in CHECKPOINTS.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = AifsDownscalerCNN(
            hidden_channels=ck["hidden_channels"],
            weather_channels=ck["weather_channels"],
        )
        model.load_state_dict(ck["model_state"])
        model = model.to(device)
        metrics = evaluate(model, dataset, entries, residual_scales, device)
        corrected = {k: round(metrics[f"{k}_corrected"], 4) for k in SHORT_NAMES}
        baseline = {k: round(metrics[f"{k}_baseline"], 4) for k in SHORT_NAMES}
        print(f"{name}: corrected {corrected}")
        print(f"{name}: weighted {metrics['weighted_corrected']:.5f} "
              f"gate {metrics['gate_mean']:.3f}")
        if name.startswith("v2"):
            print(f"baseline (linear interp): {baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
