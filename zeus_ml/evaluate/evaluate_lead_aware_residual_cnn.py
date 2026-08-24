#!/usr/bin/env python3
"""Evaluate the v2 CNN with exact streaming validator-weighted metrics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scoring import ValidatorFaithfulScorer
from evaluation.truth import Era5TruthLoader
from zeus.utils.coordinates import get_grid
from zeus.utils.region_mask import (
    OLD_REGION_CONFIGS,
    REGION_CONFIGS,
    build_geographic_weights,
)
from zeus.validator.constants import LATITUDE_WEIGHTS_PATH
from zeus_ml.datasets.lead_aware_patch_dataset import ChannelStatistics
from zeus_ml.features.era5_files import find_era5_files
from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.models.lead_aware_residual_cnn import (
    VARIABLES,
    LeadAwareGatedResidualCNN,
    build_static_features,
    context_from_cycle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split-plan",
        default=(
            "data/evaluation/plans/"
            "cnn_residual_v3_user_split.json"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test", "diagnostic"),
        default="test",
    )
    parser.add_argument("--cycle", action="append", default=[])
    parser.add_argument(
        "--bundle-root",
        default="data/evaluation/forecast_store_hist/bundles",
    )
    parser.add_argument(
        "--era5-root",
        default="data/evaluation/era5",
    )
    parser.add_argument(
        "--output",
        default=(
            "data/evaluation/results/"
            "lead_aware_residual_cnn_v2/evaluation.json"
        ),
    )
    parser.add_argument("--horizon", type=int, choices=(48, 360), default=360)
    parser.add_argument("--lead-diagnostics", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--static-root",
        default="data/evaluation/training/v4_static",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_type = checkpoint.get("model_type")
    statistics_payload = checkpoint["channel_statistics"]
    channel_statistics = ChannelStatistics.from_dict(statistics_payload)
    gfs_mean, gfs_std, residual_std = channel_statistics.tensors(device=device)
    if model_type == "lead_aware_gated_residual_cnn":
        model = LeadAwareGatedResidualCNN(**checkpoint["model_config"])
        evaluate = evaluate_cycle
        extra = {}
    elif model_type == "lead_aware_gated_residual_cnn_v4":
        from zeus_ml.models.lead_aware_residual_cnn_v4 import (
            LeadAwareGatedResidualCNNV4,
        )

        model = LeadAwareGatedResidualCNNV4(**checkpoint["model_config"])
        evaluate = evaluate_cycle_v4
        extra = {"static_root": Path(args.static_root)}
    else:
        raise SystemExit(f"Unsupported checkpoint model_type: {model_type}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    split_path = Path(args.split_plan)
    split_plan = json.loads(split_path.read_text(encoding="utf-8"))
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if args.cycle:
        cycles = tuple(args.cycle)
    else:
        cycles = tuple(split_plan[f"{args.split}_cycles"])
    if not cycles:
        raise SystemExit("No evaluation cycles selected.")

    cycle_results = []
    for cycle_key in cycles:
        cycle_result = evaluate(
            cycle_key=cycle_key,
            model=model,
            channel_statistics=(
                gfs_mean,
                gfs_std,
                residual_std,
            ),
            device=device,
            bundle_root=Path(args.bundle_root),
            era5_root=Path(args.era5_root),
            horizon=args.horizon,
            include_lead_diagnostics=args.lead_diagnostics,
            **extra,
        )
        cycle_results.append(cycle_result)
        progress = {
            "cycle": cycle_result["cycle"],
            "region_regime": cycle_result["region_regime"],
            "variables": cycle_result["variables"],
            "lead_diagnostics": len(
                cycle_result.get("lead_diagnostics", ())
            ),
        }
        print(json.dumps(progress, sort_keys=True), flush=True)

    result = {
        "schema_version": 1,
        "model_type": model_type,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_split_plan_sha256": checkpoint.get(
            "split_plan_sha256"
        ),
        "evaluation_split_plan": str(split_path.resolve()),
        "evaluation_split_plan_sha256": split_sha256,
        "benchmark_valid": bool(split_plan.get("benchmark_valid", False)),
        "split": args.split,
        "horizon_hours": args.horizon,
        "variables": list(VARIABLES),
        "cycles": cycle_results,
        "summary": summarize(cycle_results),
    }
    output_path = Path(args.output)
    _atomic_write_json(output_path, result)
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "summary": result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def evaluate_cycle(
    *,
    cycle_key: str,
    model: LeadAwareGatedResidualCNN,
    channel_statistics: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    bundle_root: Path,
    era5_root: Path,
    horizon: int,
    include_lead_diagnostics: bool = False,
) -> dict:
    cycle = datetime.strptime(cycle_key, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    bundle = bundle_root / cycle_key
    if not bundle.is_dir():
        raise FileNotFoundError(f"Missing bundle: {bundle}")
    raw_fields: list[torch.Tensor] = []
    truth_fields: list[torch.Tensor] = []
    truth_loader = Era5TruthLoader()
    for variable in VARIABLES:
        raw = load_gfs_artifact(bundle, variable, horizon)
        raw_fields.append(raw.to(torch.float16))
        del raw
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
        truth_fields.append(truth.to(torch.float16))
        del truth

    latitude_weights = torch.from_numpy(
        np.load(LATITUDE_WEIGHTS_PATH)
    ).to(torch.float32)
    latitudes = torch.linspace(-90.0, 90.0, 721)
    longitudes = torch.arange(-180.0, 180.0, 0.25)
    grid = get_grid(-90.0, 90.0, -180.0, 179.75)
    regime = ValidatorFaithfulScorer.region_regime(cycle)
    configs = (
        OLD_REGION_CONFIGS if regime == "europe_only" else REGION_CONFIGS
    )
    geographic = build_geographic_weights(grid, configs)
    metric_weights = latitude_weights[:, None] * geographic
    metric_weights = (metric_weights / metric_weights.mean()).to(device)
    static_features = build_static_features(
        latitudes,
        longitudes,
        geographic_weights=geographic,
    ).to(device)
    gfs_mean, gfs_std, residual_std = channel_statistics
    raw_accumulator = StreamingValidatorMetrics()
    corrected_accumulator = StreamingValidatorMetrics()
    gate_sums = torch.zeros(4, dtype=torch.float64)
    cells_per_channel = 0
    lead_diagnostics = []

    with torch.inference_mode():
        for lead in range(horizon + 1):
            raw = torch.stack(
                [field[lead].to(torch.float32) for field in raw_fields],
                dim=0,
            ).to(device)
            truth = torch.stack(
                [field[lead].to(torch.float32) for field in truth_fields],
                dim=0,
            ).to(device)
            model_input = (raw - gfs_mean) / gfs_std
            lead_tensor = torch.tensor(
                [float(lead)],
                dtype=torch.float32,
                device=device,
            )
            context = context_from_cycle(cycle, lead_tensor)
            output = model(
                model_input.unsqueeze(0),
                context,
                static_features,
                zonal_mean=model_input.mean(dim=-1, keepdim=False).unsqueeze(0),
                lat_starts=torch.zeros(1, dtype=torch.long, device=device),
            )
            correction = output.correction[0] * residual_std
            corrected = raw + correction
            corrected[3].clamp_(min=0.0)
            raw_accumulator.update(raw, truth, metric_weights)
            corrected_accumulator.update(corrected, truth, metric_weights)
            if include_lead_diagnostics:
                raw_lead = metrics_for_lead(raw, truth, metric_weights)
                corrected_lead = metrics_for_lead(
                    corrected,
                    truth,
                    metric_weights,
                )
                lead_diagnostics.append(
                    {
                        "lead_hour": lead,
                        "variables": {
                            variable: {
                                "raw_gfs": raw_lead[index],
                                "cnn_corrected": corrected_lead[index],
                            }
                            for index, variable in enumerate(VARIABLES)
                        },
                    }
                )
            gate_sums += output.gate[0].sum(
                dim=(-2, -1)
            ).to("cpu", dtype=torch.float64)
            cells_per_channel += output.gate.shape[-2] * output.gate.shape[-1]
            del raw, truth, model_input, output, correction, corrected

    raw_metrics = raw_accumulator.finalize()
    corrected_metrics = corrected_accumulator.finalize()
    variables = {}
    for index, variable in enumerate(VARIABLES):
        raw_metric = raw_metrics[index]
        corrected_metric = corrected_metrics[index]
        variables[variable] = {
            "raw_gfs": raw_metric,
            "cnn_corrected": corrected_metric,
            "combined_error_delta": (
                corrected_metric["combined_error"]
                - raw_metric["combined_error"]
            ),
            "combined_error_skill": (
                raw_metric["combined_error"]
                - corrected_metric["combined_error"]
            )
            / raw_metric["combined_error"],
            "cnn_wins": (
                corrected_metric["combined_error"]
                < raw_metric["combined_error"]
            ),
            "mean_gate": float(gate_sums[index] / cells_per_channel),
        }
    del raw_fields, truth_fields
    gc.collect()
    result = {
        "cycle": cycle_key,
        "region_regime": regime,
        "variables": variables,
    }
    if include_lead_diagnostics:
        result["lead_diagnostics"] = lead_diagnostics
    return result


def evaluate_cycle_v4(
    *,
    cycle_key: str,
    model,
    channel_statistics: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    bundle_root: Path,
    era5_root: Path,
    horizon: int,
    static_root: Path,
    include_lead_diagnostics: bool = False,
) -> dict:
    from zeus_ml.models.lead_aware_residual_cnn_v4 import (
        build_v4_static_features,
        cosine_solar_zenith,
        downsample_2deg,
        load_static_maps,
        valid_time_from_cycle,
    )

    cycle = datetime.strptime(cycle_key, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    bundle = bundle_root / cycle_key
    if not bundle.is_dir():
        raise FileNotFoundError(f"Missing bundle: {bundle}")
    raw_fields: list[torch.Tensor] = []
    truth_fields: list[torch.Tensor] = []
    truth_loader = Era5TruthLoader()
    for variable in VARIABLES:
        raw = load_gfs_artifact(bundle, variable, horizon)
        raw_fields.append(raw.to(torch.float16))
        del raw
        files = find_era5_files(str(era5_root), variable, cycle, horizon)
        truth = truth_loader.load(
            files,
            variable=variable,
            cycle_time=cycle,
            horizon_hours=horizon,
        ).tensor
        truth_fields.append(truth.to(torch.float16))
        del truth

    latitude_weights = torch.from_numpy(np.load(LATITUDE_WEIGHTS_PATH)).to(
        torch.float32
    )
    latitudes = torch.linspace(-90.0, 90.0, 721)
    longitudes = torch.arange(-180.0, 180.0, 0.25)
    grid = get_grid(-90.0, 90.0, -180.0, 179.75)
    regime = ValidatorFaithfulScorer.region_regime(cycle)
    configs = OLD_REGION_CONFIGS if regime == "europe_only" else REGION_CONFIGS
    geographic = build_geographic_weights(grid, configs)
    metric_weights = latitude_weights[:, None] * geographic
    metric_weights = (metric_weights / metric_weights.mean()).to(device)
    land, orography = load_static_maps(static_root)
    gfs_mean, gfs_std, residual_std = channel_statistics
    raw_accumulator = StreamingValidatorMetrics()
    corrected_accumulator = StreamingValidatorMetrics()
    gate_sums = torch.zeros(4, dtype=torch.float64)
    cells_per_channel = 0
    lead_diagnostics = []
    zeros = torch.zeros(1, dtype=torch.long, device=device)

    with torch.inference_mode():
        for lead in range(horizon + 1):
            raw = torch.stack(
                [field[lead].to(torch.float32) for field in raw_fields],
                dim=0,
            ).to(device)
            truth = torch.stack(
                [field[lead].to(torch.float32) for field in truth_fields],
                dim=0,
            ).to(device)
            zenith = cosine_solar_zenith(
                latitudes,
                longitudes,
                valid_time_from_cycle(cycle, lead),
            )
            static_features = build_v4_static_features(
                latitudes,
                longitudes,
                geographic_weights=geographic,
                land_sea=land,
                orography=orography,
                zenith=zenith,
            ).to(device)
            model_input = (raw - gfs_mean) / gfs_std
            lead_tensor = torch.tensor(
                [float(lead)],
                dtype=torch.float32,
                device=device,
            )
            context = context_from_cycle(cycle, lead_tensor)
            coarse_weather = downsample_2deg(model_input.unsqueeze(0))
            coarse_static = downsample_2deg(static_features.unsqueeze(0))
            output = model(
                model_input.unsqueeze(0),
                context,
                static_features.unsqueeze(0),
                zonal_mean=model_input.mean(dim=-1, keepdim=False).unsqueeze(0),
                lat_starts=zeros,
                coarse_input=torch.cat((coarse_weather, coarse_static), dim=1),
                lon_starts=zeros,
            )
            correction = output.correction[0] * residual_std
            corrected = raw + correction
            corrected[3].clamp_(min=0.0)
            raw_accumulator.update(raw, truth, metric_weights)
            corrected_accumulator.update(corrected, truth, metric_weights)
            if include_lead_diagnostics:
                raw_lead = metrics_for_lead(raw, truth, metric_weights)
                corrected_lead = metrics_for_lead(
                    corrected,
                    truth,
                    metric_weights,
                )
                lead_diagnostics.append(
                    {
                        "lead_hour": lead,
                        "variables": {
                            variable: {
                                "raw_gfs": raw_lead[index],
                                "cnn_corrected": corrected_lead[index],
                            }
                            for index, variable in enumerate(VARIABLES)
                        },
                    }
                )
            gate_sums += output.gate[0].sum(dim=(-2, -1)).to(
                "cpu", dtype=torch.float64
            )
            cells_per_channel += output.gate.shape[-2] * output.gate.shape[-1]
            del raw, truth, model_input, output, correction, corrected

    raw_metrics = raw_accumulator.finalize()
    corrected_metrics = corrected_accumulator.finalize()
    variables = {}
    for index, variable in enumerate(VARIABLES):
        raw_metric = raw_metrics[index]
        corrected_metric = corrected_metrics[index]
        variables[variable] = {
            "raw_gfs": raw_metric,
            "cnn_corrected": corrected_metric,
            "combined_error_delta": (
                corrected_metric["combined_error"]
                - raw_metric["combined_error"]
            ),
            "combined_error_skill": (
                raw_metric["combined_error"]
                - corrected_metric["combined_error"]
            )
            / raw_metric["combined_error"],
            "cnn_wins": (
                corrected_metric["combined_error"]
                < raw_metric["combined_error"]
            ),
            "mean_gate": float(gate_sums[index] / cells_per_channel),
        }
    del raw_fields, truth_fields
    gc.collect()
    result = {
        "cycle": cycle_key,
        "region_regime": regime,
        "variables": variables,
    }
    if include_lead_diagnostics:
        result["lead_diagnostics"] = lead_diagnostics
    return result


def metrics_for_lead(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    normalized_weights: torch.Tensor,
) -> list[dict[str, float]]:
    error = prediction - truth
    weights = normalized_weights.unsqueeze(0)
    rmse = (error.square() * weights).mean(dim=(-2, -1)).sqrt()
    mae = (error.abs() * weights).mean(dim=(-2, -1))
    return [
        {
            "rmse": float(rmse[index]),
            "mae": float(mae[index]),
            "combined_error": float((rmse[index] + mae[index]) / 2.0),
        }
        for index in range(len(VARIABLES))
    ]


class StreamingValidatorMetrics:
    """Exact time/space aggregate of the validator RMSE and MAE kernels."""

    def __init__(self) -> None:
        self.squared = torch.zeros(4, dtype=torch.float64)
        self.absolute = torch.zeros(4, dtype=torch.float64)
        self.cells = 0

    def update(
        self,
        prediction: torch.Tensor,
        truth: torch.Tensor,
        normalized_weights: torch.Tensor,
    ) -> None:
        error = prediction - truth
        weights = normalized_weights.unsqueeze(0)
        self.squared += (
            error.square() * weights
        ).sum(dim=(-2, -1)).to("cpu", dtype=torch.float64)
        self.absolute += (
            error.abs() * weights
        ).sum(dim=(-2, -1)).to("cpu", dtype=torch.float64)
        self.cells += error.shape[-2] * error.shape[-1]

    def finalize(self) -> list[dict[str, float]]:
        if self.cells < 1:
            raise RuntimeError("No metric cells accumulated.")
        mse = self.squared / self.cells
        mae = self.absolute / self.cells
        rmse = torch.sqrt(mse)
        return [
            {
                "rmse": float(rmse[index]),
                "mae": float(mae[index]),
                "combined_error": float((rmse[index] + mae[index]) / 2.0),
            }
            for index in range(4)
        ]


def summarize(cycles: list[dict]) -> dict:
    groups = {}
    for variable in VARIABLES:
        raw = [
            cycle["variables"][variable]["raw_gfs"]["combined_error"]
            for cycle in cycles
        ]
        corrected = [
            cycle["variables"][variable]["cnn_corrected"]["combined_error"]
            for cycle in cycles
        ]
        groups[variable] = {
            "cycles": len(cycles),
            "raw_mean_combined_error": statistics.fmean(raw),
            "cnn_mean_combined_error": statistics.fmean(corrected),
            "mean_combined_error_delta": statistics.fmean(
                cnn - baseline
                for cnn, baseline in zip(corrected, raw, strict=True)
            ),
            "cnn_win_count": sum(
                cnn < baseline
                for cnn, baseline in zip(corrected, raw, strict=True)
            ),
        }
    return {"cycle_count": len(cycles), "variables": groups}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    return torch.device(requested)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
