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
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zeus_ml.datasets.lead_aware_patch_dataset import (
    ChannelStatistics,
    LeadAwarePatchDataset,
    estimate_channel_statistics,
)
from zeus_ml.losses.validator_aware_residual import (
    ValidatorAwareResidualLoss,
)
from zeus_ml.models.lead_aware_residual_cnn import (
    LeadAwareGatedResidualCNN,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-plan",
        default=(
            "data/evaluation/plans/"
            "cnn_residual_v2_development_split.json"
        ),
    )
    parser.add_argument(
        "--patch-root",
        default="data/evaluation/training/lead_aware_patches",
    )
    parser.add_argument(
        "--output",
        default=(
            "data/evaluation/training/"
            "lead_aware_gated_residual_cnn_v2.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-statistics-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
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

    best_validation = float("inf")
    stale_epochs = 0
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
        append_jsonl(metrics_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

        validation_error = validation_metrics["corrected_combined_error"]
        if validation_error < best_validation:
            best_validation = validation_error
            stale_epochs = 0
            save_checkpoint(
                output_path,
                {
                    "schema_version": 2,
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
                    "epoch": epoch,
                    "best_validation_combined_error": best_validation,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "seed": args.seed,
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(
                    f"Early stopping after {epoch} epochs; "
                    f"best validation={best_validation:.6f}",
                    flush=True,
                )
                break
    print(
        json.dumps(
            {
                "checkpoint": str(output_path.resolve()),
                "metrics": str(metrics_path.resolve()),
                "best_validation_combined_error": best_validation,
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
            output = model(model_input, context, static_features)
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
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


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
