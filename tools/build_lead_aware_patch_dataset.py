#!/usr/bin/env python3
"""Build compact, explicit-split patches for lead-aware residual training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.truth import Era5TruthLoader
from evaluation.scoring import ValidatorFaithfulScorer
from zeus_ml.datasets.lead_aware_patch_dataset import sample_patch_origins
from zeus_ml.features.era5_files import find_era5_files
from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.models.lead_aware_residual_cnn import VARIABLES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-plan", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
    )
    parser.add_argument(
        "--bundle-root",
        default="data/evaluation/forecast_store_hist/bundles",
    )
    parser.add_argument(
        "--era5-root",
        default="data/evaluation/era5",
    )
    parser.add_argument(
        "--output-root",
        default="data/evaluation/training/lead_aware_patches",
    )
    parser.add_argument("--horizon", type=int, choices=(48, 360), default=360)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-lead", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan_path = Path(args.split_plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_split_plan(plan)
    cycles = tuple(plan[f"{args.split}_cycles"])
    if not cycles:
        raise SystemExit(f"No cycles configured for split {args.split!r}.")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    for cycle_offset, cycle_key in enumerate(cycles):
        build_cycle(
            cycle_key=cycle_key,
            bundle_root=Path(args.bundle_root),
            era5_root=Path(args.era5_root),
            output_root=output_root,
            horizon=args.horizon,
            patch_size=args.patch_size,
            patches_per_lead=args.patches_per_lead,
            seed=args.seed + cycle_offset,
            split=args.split,
            split_plan=str(plan_path),
            split_plan_sha256=plan_sha256,
            overwrite=args.overwrite,
        )
    print(
        json.dumps(
            {
                "split": args.split,
                "cycles": len(cycles),
                "output_root": str(output_root.resolve()),
                "split_plan_sha256": plan_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def validate_split_plan(plan: dict) -> None:
    required = ("train_cycles", "validation_cycles", "test_cycles")
    missing = [key for key in required if key not in plan]
    if missing:
        raise ValueError(f"Split plan is missing keys: {missing}")
    seen: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        cycles = plan[f"{split}_cycles"]
        if len(cycles) != len(set(cycles)):
            raise ValueError(f"Duplicate cycle in {split}_cycles.")
        for cycle in cycles:
            datetime.strptime(cycle, "%Y%m%dT%H%M%SZ")
            previous = seen.get(cycle)
            if previous is not None:
                raise ValueError(
                    f"Cycle {cycle} appears in both {previous} and {split}."
                )
            seen[cycle] = split


def build_cycle(
    *,
    cycle_key: str,
    bundle_root: Path,
    era5_root: Path,
    output_root: Path,
    horizon: int,
    patch_size: int,
    patches_per_lead: int,
    seed: int,
    split: str,
    split_plan: str,
    split_plan_sha256: str,
    overwrite: bool,
) -> None:
    bundle = bundle_root / cycle_key
    if not bundle.is_dir():
        raise FileNotFoundError(f"Missing GFS bundle: {bundle}")
    destination = output_root / cycle_key
    if destination.exists():
        if not overwrite:
            print(f"SKIP existing {destination}", flush=True)
            return
        shutil.rmtree(destination)
    temporary = output_root / f".{cycle_key}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    cycle = datetime.strptime(cycle_key, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    n_leads = horizon + 1
    full_height, full_width = 721, 1440
    if not 3 <= patch_size <= min(full_height, full_width):
        raise ValueError("patch_size is outside the supported grid range.")
    if patches_per_lead < 1:
        raise ValueError("patches_per_lead must be positive.")
    rng = np.random.default_rng(seed)
    include_germany = ValidatorFaithfulScorer.region_regime(cycle) == (
        "europe_germany"
    )
    lat_starts, lon_starts = sample_patch_origins(
        rng,
        n_leads=n_leads,
        patches_per_lead=patches_per_lead,
        patch_size=patch_size,
        include_germany=include_germany,
    )
    zonal_means = np.zeros(
        (n_leads, len(VARIABLES), full_height),
        dtype=np.float16,
    )
    shape = (
        n_leads,
        patches_per_lead,
        len(VARIABLES),
        patch_size,
        patch_size,
    )
    inputs = np.lib.format.open_memmap(
        temporary / "inputs.npy",
        mode="w+",
        dtype=np.float16,
        shape=shape,
    )
    residuals = np.lib.format.open_memmap(
        temporary / "residuals.npy",
        mode="w+",
        dtype=np.float16,
        shape=shape,
    )

    truth_loader = Era5TruthLoader()
    for variable_index, variable in enumerate(VARIABLES):
        print(f"{cycle_key} {variable}", flush=True)
        gfs = load_gfs_artifact(bundle, variable, horizon).to(torch.float32)
        files = find_era5_files(
            str(era5_root),
            variable,
            cycle,
            horizon,
        )
        truth = truth_loader.load(
            files,
            variable=variable,
            cycle_time=cycle,
            horizon_hours=horizon,
        ).tensor
        if tuple(gfs.shape) != (n_leads, full_height, full_width):
            raise ValueError(
                f"Unexpected GFS shape for {variable}: {tuple(gfs.shape)}"
            )
        zonal_means[:, variable_index] = (
            gfs.mean(dim=-1).numpy().astype(np.float16, copy=False)
        )
        if truth.shape != gfs.shape:
            raise ValueError(
                f"Truth shape {tuple(truth.shape)} does not match GFS "
                f"{tuple(gfs.shape)} for {variable}."
            )
        for lead in range(n_leads):
            for patch in range(patches_per_lead):
                lat = int(lat_starts[lead, patch])
                lon = int(lon_starts[lead, patch])
                gfs_patch = gfs[
                    lead,
                    lat : lat + patch_size,
                    lon : lon + patch_size,
                ]
                truth_patch = truth[
                    lead,
                    lat : lat + patch_size,
                    lon : lon + patch_size,
                ]
                inputs[lead, patch, variable_index] = (
                    gfs_patch.numpy().astype(np.float16, copy=False)
                )
                residuals[lead, patch, variable_index] = (
                    (truth_patch - gfs_patch)
                    .numpy()
                    .astype(np.float16, copy=False)
                )
        del gfs, truth
    inputs.flush()
    residuals.flush()
    del inputs, residuals
    np.save(
        temporary / "lead_hours.npy",
        np.arange(n_leads, dtype=np.int16),
    )
    np.save(temporary / "lat_starts.npy", lat_starts)
    np.save(temporary / "lon_starts.npy", lon_starts)
    np.save(temporary / "zonal_means.npy", zonal_means)
    metadata = {
        "schema_version": 1,
        "cycle": cycle_key,
        "split": split,
        "variables": list(VARIABLES),
        "horizon_hours": horizon,
        "full_shape": [full_height, full_width],
        "patch_size": patch_size,
        "patches_per_lead": patches_per_lead,
        "sampling": "validator_weighted_with_region_guarantees",
        "seed": seed,
        "split_plan": split_plan,
        "split_plan_sha256": split_plan_sha256,
        "region_regime": ValidatorFaithfulScorer.region_regime(cycle),
        "global_metric_weight_mean": global_metric_weight_mean(cycle),
    }
    _atomic_write_json(temporary / "metadata.json", metadata)
    os.replace(temporary, destination)
    print(f"WROTE {destination}", flush=True)


def global_metric_weight_mean(cycle: datetime) -> float:
    latitudes = torch.linspace(-90.0, 90.0, 721)
    longitudes = torch.arange(-180.0, 180.0, 0.25)
    latitude = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)[:, None]
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
    geographic = torch.ones((721, 1440))
    geographic = torch.where(europe, 1.5, geographic)
    if ValidatorFaithfulScorer.region_regime(cycle) == "europe_germany":
        geographic = torch.where(germany, 2.5, geographic)
    return float((latitude * geographic).mean())


def _atomic_write_json(path: Path, payload: dict) -> None:
    content = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
