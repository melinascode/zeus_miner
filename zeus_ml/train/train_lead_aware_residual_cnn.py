#!/usr/bin/env python3
"""Train the lead-aware gated residual CNN on explicit patch splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zeus_ml.datasets.lead_aware_patch_dataset import (
    ChannelStatistics,
    LeadAwarePatchDataset,
    estimate_channel_statistics,
)
from zeus_ml.evaluate.evaluate_lead_aware_residual_cnn import evaluate_cycle
from zeus_ml.losses.validator_aware_residual import (
    VARIABLE_WEIGHTS,
    ValidatorAwareResidualLoss,
)
from zeus_ml.models.lead_aware_residual_cnn import (
    VARIABLES,
    LeadAwareGatedResidualCNN,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-plan",
        default=(
            "data/evaluation/plans/"
            "cnn_residual_v3_user_split.json"
        ),
    )
    parser.add_argument(
        "--patch-root",
        default="data/evaluation/training/lead_aware_patches_v3",
    )
    parser.add_argument(
        "--output",
        default=(
            "data/evaluation/training/"
            "lead_aware_gated_residual_cnn_v3.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-statistics-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--selection-every", type=int, default=3)
    parser.add_argument("--selection-max-cycles", type=int, default=4)
    parser.add_argument(
        "--bundle-root",
        default="data/evaluation/forecast_store_hist/bundles",
    )
    parser.add_argument(
        "--era5-root",
        default="data/evaluation/era5",
    )
    parser.add_argument("--horizon", type=int, default=360)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive.")
    set_seed(args.seed)
    device = choose_device(args.device)
    split_path = Path(args.split_plan)
    split_plan = json.loads(split_path.read_text(encoding="utf-8"))
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()
    train_cycles = tuple(split_plan["train_cycles"])
    validation_cycles = tuple(split_plan["validation_cycles"])
    if not train_cycles or not validation_cycles:
        raise SystemExit("Training and validation splits must both be non-empty.")
    patch_root = Path(args.patch_root)

    raw_train = LeadAwarePatchDataset(
        patch_root,
        cycles=train_cycles,
    )
    statistics = estimate_channel_statistics(
        raw_train,
        max_samples=args.max_statistics_samples,
        seed=args.seed,
    )
    train_dataset = LeadAwarePatchDataset(
        patch_root,
        cycles=train_cycles,
        statistics=statistics,
    )
    validation_dataset = LeadAwarePatchDataset(
        patch_root,
        cycles=validation_cycles,
        statistics=statistics,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        sampler=WeightedRandomSampler(
            weights=train_dataset.sample_weights(),
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        ),
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    model_config = {
        "hidden_channels": args.hidden_channels,
        "dilations": [1, 2, 4, 8, 4, 2],
        "dropout": 0.05,
        "initial_gate": 0.02,
    }
    model = LeadAwareGatedResidualCNN(**model_config).to(device)
    criterion = ValidatorAwareResidualLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )
    _, _, residual_scales = statistics.tensors(device=device)
    residual_scales = residual_scales.unsqueeze(0)
    output_path = Path(args.output)
    metrics_path = output_path.with_suffix(".metrics.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_selection = float("inf")
    stale_epochs = 0
    selection_cycles = tuple(validation_cycles[: args.selection_max_cycles])
    gfs_mean, gfs_std, residual_std = statistics.tensors(device=device)
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            residual_scales=residual_scales,
            device=device,
            optimizer=optimizer,
        )
        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            residual_scales=residual_scales,
            device=device,
            optimizer=None,
        )
        scheduler.step(validation_metrics["corrected_combined_error"])
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        run_selection = (
            epoch == 1
            or epoch == args.epochs
            or (
                args.selection_every > 0
                and epoch % args.selection_every == 0
            )
        )
        if run_selection:
            selection = run_validator_selection(
                model=model,
                channel_statistics=(gfs_mean, gfs_std, residual_std),
                cycles=selection_cycles,
                device=device,
                bundle_root=Path(args.bundle_root),
                era5_root=Path(args.era5_root),
                horizon=args.horizon,
            )
            record["validator_selection"] = selection
        append_jsonl(metrics_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

        if not run_selection:
            continue
        selection_error = record["validator_selection"]["mean_relative_combined_error"]
        if selection_error < best_selection:
            best_selection = selection_error
            stale_epochs = 0
            save_checkpoint(
                output_path,
                {
                    "schema_version": 3,
                    "model_type": "lead_aware_gated_residual_cnn",
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "channel_statistics": statistics.as_dict(),
                    "split_plan": str(split_path),
                    "split_plan_sha256": split_sha256,
                    "benchmark_valid": bool(
                        split_plan.get("benchmark_valid", False)
                    ),
                    "train_cycles": list(train_cycles),
                    "validation_cycles": list(validation_cycles),
                    "selection_cycles": list(selection_cycles),
                    "epoch": epoch,
                    "best_validator_relative_combined_error": best_selection,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "seed": args.seed,
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(
                    f"Early stopping after {epoch} epochs; "
                    f"best validator selection={best_selection:.6f}",
                    flush=True,
                )
                break
    print(
        json.dumps(
            {
                "checkpoint": str(output_path.resolve()),
                "metrics": str(metrics_path.resolve()),
                "best_validator_relative_combined_error": best_selection,
                "device": str(device),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_epoch(
    *,
    model: LeadAwareGatedResidualCNN,
    loader: DataLoader,
    criterion: ValidatorAwareResidualLoss,
    residual_scales: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    samples = 0
    for batch in loader:
        model_input = batch["model_input"].to(device)
        context = batch["context"].to(device)
        static_features = batch["static_features"].to(device)
        raw = batch["raw"].to(device)
        truth = batch["truth"].to(device)
        metric_weights = batch["metric_weights"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                model_input,
                context,
                static_features,
                zonal_mean=batch["zonal_mean"].to(device),
                lat_starts=batch["lat_start"].to(device),
            )
            loss_output = criterion(
                raw_forecast=raw,
                truth=truth,
                correction=output.correction,
                gate=output.gate,
                metric_weights=metric_weights,
                residual_scales=residual_scales,
            )
            if training:
                loss_output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        batch_size = model_input.shape[0]
        samples += batch_size
        for key, value in loss_output.detached_metrics().items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
    if samples < 1:
        raise RuntimeError("Data loader produced no samples.")
    return {key: value / samples for key, value in totals.items()}


def make_loader(
    dataset: LeadAwarePatchDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def run_validator_selection(
    *,
    model: LeadAwareGatedResidualCNN,
    channel_statistics: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cycles: tuple[str, ...],
    device: torch.device,
    bundle_root: Path,
    era5_root: Path,
    horizon: int,
) -> dict:
    if not cycles:
        raise RuntimeError("Validator selection requires at least one cycle.")
    was_training = model.training
    model.eval()
    cycle_scores = []
    variable_ratios: dict[str, list[float]] = {name: [] for name in VARIABLES}
    for cycle_key in cycles:
        result = evaluate_cycle(
            cycle_key=cycle_key,
            model=model,
            channel_statistics=channel_statistics,
            device=device,
            bundle_root=bundle_root,
            era5_root=era5_root,
            horizon=horizon,
        )
        weighted = 0.0
        for weight, variable in zip(VARIABLE_WEIGHTS, VARIABLES, strict=True):
            raw = result["variables"][variable]["raw_gfs"]["combined_error"]
            cnn = result["variables"][variable]["cnn_corrected"]["combined_error"]
            ratio = cnn / raw
            variable_ratios[variable].append(ratio)
            weighted += float(weight) * ratio
        cycle_scores.append(
            {
                "cycle": cycle_key,
                "mean_relative_combined_error": weighted,
                "variables": result["variables"],
            }
        )
        print(
            json.dumps(
                {
                    "validator_selection_cycle": cycle_key,
                    "mean_relative_combined_error": weighted,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    model.train(was_training)
    return {
        "cycles": cycle_scores,
        "mean_relative_combined_error": float(np.mean(
            [row["mean_relative_combined_error"] for row in cycle_scores]
        )),
        "variable_mean_relative_combined_error": {
            variable: float(np.mean(values))
            for variable, values in variable_ratios.items()
        },
    }


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
