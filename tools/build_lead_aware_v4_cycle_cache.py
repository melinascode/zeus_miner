#!/usr/bin/env python3
"""Cache native 0.25° GFS and ERA5 residual cubes for v4 tile training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.truth import Era5TruthLoader
from zeus_ml.features.era5_files import find_era5_files
from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.models.lead_aware_residual_cnn import VARIABLES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-plan",
        default="data/evaluation/plans/cnn_residual_v4_seasonal_split.json",
    )
    parser.add_argument(
        "--splits",
        default="train,validation",
        help="Comma-separated split names to cache.",
    )
    parser.add_argument(
        "--output-root",
        default="data/evaluation/training/lead_aware_v4_cycles",
    )
    parser.add_argument(
        "--bundle-root",
        default="data/evaluation/forecast_store_hist/bundles",
    )
    parser.add_argument("--era5-root", default="data/evaluation/era5")
    parser.add_argument("--horizon", type=int, default=360)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    split_path = Path(args.split_plan)
    split_plan = json.loads(split_path.read_text(encoding="utf-8"))
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    names = tuple(
        part.strip() for part in args.splits.split(",") if part.strip()
    )
    cycles = []
    for name in names:
        cycles.extend(split_plan[f"{name}_cycles"])
    seen = []
    for cycle in cycles:
        if cycle not in seen:
            seen.append(cycle)
    truth_loader = Era5TruthLoader()
    for cycle_key in seen:
        destination = output_root / cycle_key
        if (destination / "metadata.json").is_file() and not args.overwrite:
            print(f"SKIP {cycle_key}", flush=True)
            continue
        print(f"CACHE {cycle_key}", flush=True)
        _write_cycle(
            cycle_key=cycle_key,
            destination=destination,
            bundle_root=Path(args.bundle_root),
            era5_root=Path(args.era5_root),
            horizon=args.horizon,
            split_plan=str(split_path),
            split_sha=split_sha,
            truth_loader=truth_loader,
        )
    print({"cached_cycles": seen, "output": str(output_root.resolve())})
    return 0


def _write_cycle(
    *,
    cycle_key: str,
    destination: Path,
    bundle_root: Path,
    era5_root: Path,
    horizon: int,
    split_plan: str,
    split_sha: str,
    truth_loader: Era5TruthLoader,
) -> None:
    cycle = datetime.strptime(cycle_key, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    n_leads = horizon + 1
    inputs = np.zeros((n_leads, 4, 721, 1440), dtype=np.float16)
    residuals = np.zeros_like(inputs)
    zonal = np.zeros((n_leads, 4, 721), dtype=np.float16)
    bundle = bundle_root / cycle_key
    for index, variable in enumerate(VARIABLES):
        gfs = load_gfs_artifact(bundle, variable, horizon).to(torch.float32)
        files = find_era5_files(str(era5_root), variable, cycle, horizon)
        truth = truth_loader.load(
            files,
            variable=variable,
            cycle_time=cycle,
            horizon_hours=horizon,
        ).tensor
        residual = (truth - gfs).to(torch.float32)
        inputs[:, index] = gfs.numpy().astype(np.float16, copy=False)
        residuals[:, index] = residual.numpy().astype(np.float16, copy=False)
        zonal[:, index] = gfs.mean(dim=-1).numpy().astype(np.float16, copy=False)
        del gfs, truth, residual
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        import shutil

        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    np.save(temporary / "inputs.npy", inputs)
    np.save(temporary / "residuals.npy", residuals)
    np.save(temporary / "zonal_means.npy", zonal)
    metadata = {
        "schema_version": 1,
        "cycle": cycle_key,
        "variables": list(VARIABLES),
        "shape": [n_leads, 4, 721, 1440],
        "split_plan": split_plan,
        "split_plan_sha256": split_sha,
    }
    (temporary / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        import shutil

        shutil.rmtree(destination)
    os.replace(temporary, destination)


if __name__ == "__main__":
    raise SystemExit(main())
