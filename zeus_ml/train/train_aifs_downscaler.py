"""Train the hourly AIFS downscaler against ERA5 truth.

The model corrects linearly interpolated 6-hourly AIFS Single forecasts onto
the hourly grid Zeus scores. Validation runs on the full 721x1440 globe so the
reported numbers are the ones the subnet will actually see, and every metric is
quoted against the linear-interpolation baseline the model must beat.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from zeus_ml.datasets.aifs_downscale_dataset import (
    AifsDownscaleDataset,
    CycleBlockSampler,
    Era5HourlyReader,
    estimate_statistics,
)
from zeus_ml.losses.validator_aware_residual import ValidatorAwareResidualLoss
from zeus_ml.models.aifs_downscaler_cnn import (
    MAX_LEAD_HOURS,
    VARIABLE_WEIGHTS,
    VARIABLES,
    AifsDownscalerCNN,
    DownscalerStatistics,
)


AIFS_ROOT = "/Zeus/data/evaluation/aifs_single"
ERA5_ROOT = "/Zeus/data/evaluation/era5"
STATIC_ROOT = "/Zeus/data/evaluation/training/aifs_static"
OUTPUT_ROOT = "/Zeus/data/evaluation/training"

# One holdout block per season. Forecasts run 15 days, so training cycles
# within 15 days of a block are dropped: otherwise a training cycle and a
# validation cycle would be scored against overlapping ERA5 valid times.
VALIDATION_BLOCKS = (
    ("2025-07-15", "2025-07-28"),
    ("2025-10-15", "2025-10-28"),
    ("2026-01-14", "2026-01-27"),
    ("2026-04-08", "2026-04-21"),
)
BUFFER_DAYS = 15
SHORT_NAMES = ("t2m", "u100", "v100")


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    validation: tuple[str, ...]


def build_split(cycles: list[str]) -> Split:
    def as_date(key: str) -> datetime:
        return datetime.strptime(key, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)

    blocks = [
        (
            datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
            datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
        )
        for start, end in VALIDATION_BLOCKS
    ]
    train, validation = [], []
    for key in cycles:
        date = as_date(key)
        if any(start <= date <= end for start, end in blocks):
            validation.append(key)
        elif not any(
            start - timedelta(days=BUFFER_DAYS)
            <= date
            <= end + timedelta(days=BUFFER_DAYS)
            for start, end in blocks
        ):
            train.append(key)
    if not validation:
        # ERA5 coverage misses every seasonal block (e.g. a summer-only
        # slice). Fall back to a chronological holdout: newest ~15% of
        # cycles validate, training keeps a 15-day buffer before them.
        ordered = sorted(cycles, key=as_date)
        n_val = max(2, len(ordered) // 7)
        validation = ordered[-n_val:]
        cutoff = as_date(validation[0]) - timedelta(days=BUFFER_DAYS)
        train = [key for key in ordered[:-n_val] if as_date(key) <= cutoff]
        print(
            f"no cycles inside the seasonal validation blocks; using a "
            f"chronological holdout of the newest {n_val} cycles instead",
            flush=True,
        )
    return Split(train=tuple(train), validation=tuple(validation))


def combined_error(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    metric_weights: torch.Tensor,
) -> torch.Tensor:
    """Validator combined error (iwRMSE + iwMAE) / 2, per variable."""

    error = prediction - truth
    rmse = torch.sqrt((error.square() * metric_weights).mean(dim=(-2, -1)))
    mae = (error.abs() * metric_weights).mean(dim=(-2, -1))
    return (rmse + mae) / 2.0


def validation_entries(n_cycles: int, per_cycle: int, seed: int = 5) -> list[tuple]:
    """Deterministic (cycle, lead) pairs covering the full lead range."""

    rng = np.random.default_rng(seed)
    entries = []
    for cycle in range(n_cycles):
        # Stratify leads so every cycle contributes short, medium and long
        # forecasts, and offset by cycle so all diurnal phases get covered.
        edges = np.linspace(0, MAX_LEAD_HOURS, per_cycle + 1)
        for i in range(per_cycle):
            lead = int(rng.integers(int(edges[i]), int(edges[i + 1]) + 1))
            entries.append((cycle, min(lead, MAX_LEAD_HOURS)))
    return entries


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: AifsDownscaleDataset,
    entries: list[tuple],
    residual_scales: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    variable_weights: tuple[float, ...] = VARIABLE_WEIGHTS,
) -> dict[str, float]:
    model.eval()
    corrected_sum = np.zeros(len(VARIABLES))
    raw_sum = np.zeros(len(VARIABLES))
    gate_sum = 0.0
    # Half-open bands; lead 72 belongs to the long window it is selected on.
    bands = {
        "h0_72": (0, 72),
        "h72_360": (72, 361),
    }
    band_corr = {name: np.zeros(len(VARIABLES)) for name in bands}
    band_raw = {name: np.zeros(len(VARIABLES)) for name in bands}
    band_n = {name: 0 for name in bands}
    for cycle, lead in entries:
        item = dataset.full_globe_item(cycle, lead)
        item = {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in item.items()}
        with torch.no_grad():
            output = model(
                item["model_input"], item["context"], item["static_features"]
            )
        corrected = item["raw"] + output.correction * residual_scales
        corr = (
            combined_error(corrected, item["truth"], item["metric_weights"])
            .squeeze(0)
            .cpu()
            .numpy()
        )
        raw = (
            combined_error(item["raw"], item["truth"], item["metric_weights"])
            .squeeze(0)
            .cpu()
            .numpy()
        )
        corrected_sum += corr
        raw_sum += raw
        gate_sum += float(output.gate.mean())
        for name, (lo, hi) in bands.items():
            if lo <= lead < hi:
                band_corr[name] += corr
                band_raw[name] += raw
                band_n[name] += 1
    model.train()
    n = len(entries)
    corrected_mean = corrected_sum / n
    raw_mean = raw_sum / n
    weights = np.asarray(variable_weights, dtype=np.float64)
    metrics = {"gate_mean": gate_sum / n}
    for i, key in enumerate(SHORT_NAMES):
        metrics[f"{key}_corrected"] = float(corrected_mean[i])
        metrics[f"{key}_baseline"] = float(raw_mean[i])
        metrics[f"{key}_gain_pct"] = float(
            100.0 * (raw_mean[i] - corrected_mean[i]) / max(raw_mean[i], 1e-9)
        )
    metrics["weighted_corrected"] = float((corrected_mean / raw_mean * weights).sum())
    metrics["weighted_gain_pct"] = float(
        100.0 * (1.0 - (corrected_mean / raw_mean * weights).sum() / weights.sum())
    )
    metrics["wind_corrected"] = 0.5 * (
        metrics["u100_corrected"] + metrics["v100_corrected"]
    )
    for name, count in band_n.items():
        if count == 0:
            continue
        cmean = band_corr[name] / count
        rmean = band_raw[name] / count
        metrics[f"{name}_n"] = int(count)
        for i, key in enumerate(SHORT_NAMES):
            metrics[f"{name}_{key}_corrected"] = float(cmean[i])
            metrics[f"{name}_{key}_gain_pct"] = float(
                100.0 * (rmean[i] - cmean[i]) / max(rmean[i], 1e-9)
            )
        metrics[f"{name}_wind_corrected"] = 0.5 * float(cmean[1] + cmean[2])
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aifs-root", default=AIFS_ROOT)
    parser.add_argument(
        "--ens-root",
        default=None,
        help="Train on ENS-mean .npy cycles; the lagged slot then carries the "
        "same-day AIFS Single run.",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Warm-start model weights from this checkpoint.",
    )
    parser.add_argument(
        "--drop-after",
        default=None,
        help="Exclude cycles after this ISO date (protects held-out eval truth).",
    )
    parser.add_argument("--era5-root", default=ERA5_ROOT)
    parser.add_argument("--static-root", default=STATIC_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--name", default="aifs_downscaler_v1")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--cycles-per-epoch", type=int, default=96)
    parser.add_argument("--leads-per-cycle", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--tiles-per-item", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument(
        "--use-lagged",
        action="store_true",
        help="Feed yesterday's run at the same valid time as extra channels.",
    )
    parser.add_argument(
        "--use-temporal-mix",
        action="store_true",
        help="Add the fraction-aware temporal head: a zero-init 4f(1-f)-gated "
        "term on the 6h tendency, so mid-hours are predicted from both "
        "anchors instead of pure linear interpolation.",
    )
    parser.add_argument(
        "--mix-head-only",
        action="store_true",
        help="Freeze the V3 trunk and train only mix_heads (requires "
        "--use-temporal-mix). Isolates the mid-hour interpolator.",
    )
    parser.add_argument(
        "--statistics-from",
        default=None,
        help="Reuse a statistics JSON from a previous run.",
    )
    parser.add_argument(
        "--geo-mode",
        choices=("boxes", "official"),
        default="boxes",
        help="Loss metric weights: Europe/Germany boxes (pre-2026-08-25 rules) "
        "or the official per-variable capacity scalars.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-regret-weight", type=float, default=0.5)
    parser.add_argument(
        "--mae-weight",
        type=float,
        default=0.5,
        help="Weight of the MAE term inside the combined loss "
        "(0.5 = validator metric; >0.5 tilts toward MAE).",
    )
    parser.add_argument(
        "--midhour-boost",
        type=float,
        default=0.0,
        help="Oversample leads off the 6h steps by this factor.",
    )
    parser.add_argument(
        "--gradient-weight",
        type=float,
        default=0.0,
        help="Weight of the spatial-gradient sharpness term in the loss "
        "(0 disables; attacks the ~80%% gradient-energy smoothing).",
    )
    parser.add_argument(
        "--long-lead-boost",
        type=float,
        default=0.0,
        help="Oversample leads past --long-lead-start by this factor "
        "(the 361h window carries 80%% of the incentive).",
    )
    parser.add_argument("--long-lead-start", type=int, default=72)
    parser.add_argument(
        "--speed-weight",
        type=float,
        default=0.0,
        help="Weight of the 100m wind-speed MAE term (0 disables; counters "
        "the ensemble-mean |V| damping that u/v errors alone do not see).",
    )
    parser.add_argument(
        "--variable-weights",
        default=None,
        help="Comma-separated loss/selection weights for 2t,100u,100v "
        f"(default {','.join(str(w) for w in VARIABLE_WEIGHTS)}; "
        "e.g. 0.2,0.4,0.4 to tilt selection toward wind).",
    )
    parser.add_argument(
        "--wind-no-regret-weight",
        type=float,
        default=0.0,
        help="Extra no-regret on 100u/100v only (stops V3-style wind regression).",
    )
    parser.add_argument(
        "--select-on",
        choices=("weighted", "wind", "wind_long"),
        default="weighted",
        help="Checkpoint on full weighted error, mean 100u/100v, or 100u/100v "
        "for leads >= 72h (360h wind).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--min-cycles",
        type=int,
        default=8,
        help="Minimum usable ENS cycles with full ERA5 windows (lower only "
        "for smoke tests).",
    )
    parser.add_argument("--europe-fraction", type=float, default=0.35)
    parser.add_argument("--validation-cycles", type=int, default=12)
    parser.add_argument("--validation-per-cycle", type=int, default=6)
    parser.add_argument("--stat-items", type=int, default=48)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--grib-cache-size",
        type=int,
        default=None,
        help="Decoded AIFS-single cycles kept in RAM (default 3 with "
        "--use-lagged). Raise to the train-set size to avoid re-decoding.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mix_head_only and not args.use_temporal_mix:
        raise SystemExit("--mix-head-only requires --use-temporal-mix")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / f"{args.name}.pt"
    metrics_path = output_root / f"{args.name}.metrics.jsonl"

    if args.ens_root:
        all_cycles = sorted(p.stem for p in Path(args.ens_root).glob("*.npy"))
    else:
        all_cycles = sorted(p.stem for p in Path(args.aifs_root).glob("*.grib2"))
    era5_reader = Era5HourlyReader(args.era5_root)
    kept = []
    skipped = 0
    for key in all_cycles:
        cycle_time = datetime.strptime(key, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
        days_ok = all(
            era5_reader.has(cycle_time + timedelta(hours=hour))
            for hour in range(0, MAX_LEAD_HOURS + 1, 24)
        )
        if days_ok:
            kept.append(key)
        else:
            skipped += 1
    all_cycles = kept
    print(
        f"ERA5 coverage: {len(all_cycles)} usable cycles, {skipped} skipped",
        flush=True,
    )
    if len(all_cycles) < args.min_cycles:
        raise SystemExit(
            "Not enough cycles with full ERA5 windows to train. "
            "Fetch more ERA5 days and retry."
        )
    if args.drop_after:
        limit = datetime.fromisoformat(args.drop_after).replace(tzinfo=timezone.utc)
        all_cycles = [
            key
            for key in all_cycles
            if datetime.strptime(key, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            <= limit
        ]
    split = build_split(all_cycles)
    print(
        f"cycles: {len(all_cycles)} total -> {len(split.train)} train, "
        f"{len(split.validation)} validation (seasonal blocks, "
        f"{BUFFER_DAYS}-day buffer)",
        flush=True,
    )

    init_payload = None
    if args.init_from:
        init_payload = torch.load(
            args.init_from, map_location="cpu", weights_only=False
        )
        ckpt_hidden = init_payload.get("hidden_channels")
        if ckpt_hidden is not None and int(ckpt_hidden) != args.hidden_channels:
            print(
                f"overriding --hidden-channels {args.hidden_channels} -> "
                f"{ckpt_hidden} from {args.init_from}",
                flush=True,
            )
            args.hidden_channels = int(ckpt_hidden)
        if init_payload.get("use_lagged") and not args.use_lagged:
            print("enabling --use-lagged to match checkpoint", flush=True)
            args.use_lagged = True
        if init_payload.get("use_temporal_mix") and not args.use_temporal_mix:
            print("enabling --use-temporal-mix to match checkpoint", flush=True)
            args.use_temporal_mix = True

    common = dict(
        aifs_root=args.aifs_root,
        ens_root=args.ens_root,
        era5_root=args.era5_root,
        static_root=args.static_root,
        tile_size=args.tile_size,
        tiles_per_item=args.tiles_per_item,
        europe_fraction=args.europe_fraction,
        use_lagged=args.use_lagged,
        grib_cache_size=(
            args.grib_cache_size
            if args.grib_cache_size is not None
            else (3 if args.use_lagged else 2)
        ),
        geo_mode=args.geo_mode,
    )
    stats_path = output_root / f"{args.name}.statistics.json"
    if args.statistics_from and not stats_path.is_file():
        stats_path.write_text(
            Path(args.statistics_from).read_text(encoding="utf-8"), encoding="utf-8"
        )
    if (
        init_payload is not None
        and isinstance(init_payload.get("statistics"), dict)
        and not stats_path.is_file()
        and not args.statistics_from
    ):
        stats_path.write_text(
            json.dumps(init_payload["statistics"], indent=2), encoding="utf-8"
        )
    if stats_path.is_file():
        statistics = DownscalerStatistics.from_dict(
            json.loads(stats_path.read_text(encoding="utf-8"))
        )
        print("loaded statistics", statistics.to_dict(), flush=True)
    elif init_payload is not None and isinstance(init_payload.get("statistics"), dict):
        statistics = DownscalerStatistics.from_dict(init_payload["statistics"])
        stats_path.write_text(json.dumps(statistics.to_dict(), indent=2), "utf-8")
        print("loaded statistics from checkpoint", statistics.to_dict(), flush=True)
    else:
        print(f"estimating statistics from {args.stat_items} cycles ...", flush=True)
        raw_dataset = AifsDownscaleDataset(
            cycles=split.train, seed=args.seed, **common
        )
        statistics = estimate_statistics(raw_dataset, n_items=args.stat_items)
        stats_path.write_text(json.dumps(statistics.to_dict(), indent=2), "utf-8")
        print("statistics", statistics.to_dict(), flush=True)
        del raw_dataset

    train_dataset = AifsDownscaleDataset(
        cycles=split.train, statistics=statistics, seed=args.seed, **common
    )
    val_dataset = AifsDownscaleDataset(
        cycles=split.validation, statistics=statistics, seed=args.seed + 1, **common
    )
    sampler = CycleBlockSampler(
        train_dataset,
        cycles_per_epoch=args.cycles_per_epoch,
        leads_per_cycle=args.leads_per_cycle,
        seed=args.seed,
        midhour_boost=args.midhour_boost,
        long_lead_boost=args.long_lead_boost,
        long_lead_start=args.long_lead_start,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=0,
    )
    val_entries = validation_entries(
        min(args.validation_cycles, len(val_dataset.cycles)),
        args.validation_per_cycle,
    )

    weather_channels = 12 if args.use_lagged else 6
    model = AifsDownscalerCNN(
        hidden_channels=args.hidden_channels,
        weather_channels=weather_channels,
        use_temporal_mix=args.use_temporal_mix,
    )
    if init_payload is not None:
        missing, unexpected = model.load_state_dict(
            init_payload["model_state"], strict=False
        )
        # Warm-starting a temporal-mix model from a mix-less checkpoint
        # (e.g. V3) legitimately leaves the mix heads at their zero init;
        # anything else missing or unexpected is a real mismatch.
        fresh = {key for key in missing if key.startswith("mix_heads.")}
        if unexpected or set(missing) - fresh:
            raise RuntimeError(
                f"checkpoint mismatch missing={sorted(set(missing) - fresh)} "
                f"unexpected={list(unexpected)}"
            )
        if fresh:
            print(
                f"temporal-mix heads not in checkpoint; keeping "
                f"{len(fresh)} tensors at zero init (starts as exact "
                "linear interpolation)",
                flush=True,
            )
        print(
            f"warm-started from {args.init_from} "
            f"(epoch {init_payload.get('epoch')}, "
            f"hidden={args.hidden_channels}, "
            f"weather={weather_channels}, "
            f"temporal_mix={args.use_temporal_mix})",
            flush=True,
        )
    n_params = sum(p.numel() for p in model.parameters())
    device = torch.device(args.device)
    model = model.to(device)
    trainable = list(model.parameters())
    if args.mix_head_only:
        if model.mix_heads is None:
            raise RuntimeError("mix-head-only requested but mix_heads is missing")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.mix_heads.parameters():
            param.requires_grad = True
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        print(
            f"mix-head-only: frozen trunk, {n_train} trainable / {n_params} total",
            flush=True,
        )
    print(f"model: {n_params} parameters, tile {args.tile_size}, "
          f"device {device}", flush=True)
    variable_weights = tuple(VARIABLE_WEIGHTS)
    if args.variable_weights:
        parsed = tuple(float(x) for x in args.variable_weights.split(","))
        if len(parsed) != len(VARIABLES) or min(parsed) < 0.0 or sum(parsed) <= 0:
            raise ValueError(
                f"--variable-weights needs {len(VARIABLES)} non-negative values"
            )
        variable_weights = parsed
        print(f"variable weights: {variable_weights}", flush=True)
    wind_channels = (
        VARIABLES.index("100m_u_component_of_wind"),
        VARIABLES.index("100m_v_component_of_wind"),
    )
    criterion = ValidatorAwareResidualLoss(
        variable_weights=variable_weights,
        solar_channel=None,
        no_regret_weight=args.no_regret_weight,
        mae_weight=args.mae_weight,
        gradient_weight=args.gradient_weight,
        speed_weight=args.speed_weight,
        wind_channels=wind_channels,
        wind_no_regret_weight=args.wind_no_regret_weight,
        gradient_wind_only=True,
    ).to(device)
    residual_scales = torch.tensor(statistics.residual_std, dtype=torch.float32).view(
        len(VARIABLES), 1, 1
    ).to(device)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = len(sampler)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=0.25,
    )

    print(
        f"validating baseline on {len(val_entries)} full-globe samples ...",
        flush=True,
    )
    baseline = evaluate(
        model, val_dataset, val_entries, residual_scales, device,
        variable_weights=variable_weights,
    )
    print(
        "  linear interpolation combined error  "
        + "  ".join(f"{k}={baseline[f'{k}_baseline']:.4f}" for k in SHORT_NAMES),
        flush=True,
    )

    best = math.inf
    with metrics_path.open("a", encoding="utf-8") as log:
        log.write(
            json.dumps({"event": "baseline", "metrics": baseline, "params": n_params})
            + "\n"
        )
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            t_epoch = time.time()
            running = {"loss": 0.0, "no_regret": 0.0, "gate": 0.0}
            for step, item in enumerate(loader, start=1):
                item = {
                    k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in item.items()
                }
                output = model(
                    item["model_input"], item["context"], item["static_features"]
                )
                result = criterion(
                    raw_forecast=item["raw"],
                    truth=item["truth"],
                    correction=output.correction,
                    gate=output.gate,
                    metric_weights=item["metric_weights"],
                    residual_scales=residual_scales,
                )
                optimizer.zero_grad(set_to_none=True)
                result.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running["loss"] += float(result.loss.detach())
                running["no_regret"] += float(result.no_regret_penalty.detach())
                running["gate"] += float(output.gate.mean().detach())
                if step % 100 == 0:
                    print(
                        f"  epoch {epoch} step {step}/{steps_per_epoch} "
                        f"loss={running['loss']/step:.4f} "
                        f"no_regret={running['no_regret']/step:.4f} "
                        f"gate={running['gate']/step:.3f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"({(time.time()-t_epoch)/step:.2f}s/step)",
                        flush=True,
                    )
            metrics = evaluate(
                model, val_dataset, val_entries, residual_scales, device,
                variable_weights=variable_weights,
            )
            metrics["epoch"] = epoch
            metrics["train_loss"] = running["loss"] / max(steps_per_epoch, 1)
            metrics["seconds"] = time.time() - t_epoch
            log.write(json.dumps({"event": "epoch", "metrics": metrics}) + "\n")
            log.flush()
            print(
                f"epoch {epoch}: train_loss={metrics['train_loss']:.4f} "
                f"gain t2m={metrics['t2m_gain_pct']:+.2f}% "
                f"u100={metrics['u100_gain_pct']:+.2f}% "
                f"v100={metrics['v100_gain_pct']:+.2f}% "
                f"weighted={metrics['weighted_gain_pct']:+.2f}% "
                f"wind={metrics['wind_corrected']:.4f} "
                f"wind72-360={metrics.get('h72_360_wind_corrected', float('nan')):.4f} "
                f"gate={metrics['gate_mean']:.3f} "
                f"({metrics['seconds']/60:.1f} min)",
                flush=True,
            )
            if args.select_on == "wind":
                score = metrics["wind_corrected"]
            elif args.select_on == "wind_long":
                score = metrics.get("h72_360_wind_corrected", metrics["wind_corrected"])
            else:
                score = metrics["weighted_corrected"]
            if score < best:
                best = score
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "statistics": statistics.to_dict(),
                        "variables": list(VARIABLES),
                        "variable_weights": list(variable_weights),
                        "hidden_channels": args.hidden_channels,
                        "weather_channels": weather_channels,
                        "use_lagged": args.use_lagged,
                        "use_temporal_mix": args.use_temporal_mix,
                        "mix_head_only": args.mix_head_only,
                        "ens_mode": bool(args.ens_root),
                        "geo_mode": args.geo_mode,
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                    checkpoint_path,
                )
                print(f"  saved checkpoint (weighted={score:.5f})", flush=True)
    print(f"done. best weighted score {best:.5f} -> {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
