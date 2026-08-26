"""Train the Europe ResUNet specialist on preprocessed Europe crops.

Two phases, same script:

  pretrain   --source single   483 daily AIFS-Single cycles (~29k samples)
  finetune   --source ens      weekly ENS-mean cycles, warm-started

Loss is the official validator metric restricted to the crop: cos-lat x
capacity scalar (temperature map for t2m, wind map for u/v), optionally masked
to the Germany box (--loss-region germany).

Runs on CPU for smoke tests and on CUDA with AMP for real training. Needs only
torch + numpy + the europe_crops directory produced by
tools/preprocess_europe_crops.py.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from zeus_ml.losses.validator_aware_residual import ValidatorAwareResidualLoss
from zeus_ml.models.europe_resunet import (
    IN_CHANNELS,
    MAX_LEAD_HOURS,
    VARIABLE_WEIGHTS,
    EuropeResUNet,
    bracket_for_lead,
    build_context,
    cosine_solar_zenith,
    evaluate_climatology,
)
from zeus_ml.models.germany_resunet import (
    GERMANY_HEIGHT,
    GERMANY_LAT_SLICE,
    GERMANY_LON_SLICE,
    GERMANY_WIDTH,
    GermanyResUNet,
)

SHORT_NAMES = ("t2m", "u100", "v100")
VALIDATION_BLOCKS = (
    ("2025-07-15", "2025-07-28"),
    ("2025-10-15", "2025-10-28"),
    ("2026-01-14", "2026-01-27"),
    ("2026-04-08", "2026-04-21"),
)
BUFFER_DAYS = 15
GERMANY_LAT = (47.0, 56.0)
GERMANY_LON = (6.0, 15.0)
# Zeus ladder weights: 48h challenge 0.2, 360h challenge 0.8. A lead <= 48
# is scored by both ladders, so its expected incentive weight is higher.
LADDER_SHORT_WEIGHT = 0.2
LADDER_LONG_WEIGHT = 0.8
SHORT_MAX_LEAD = 48


def lead_probabilities() -> np.ndarray:
    """P(lead) proportional to how much incentive mass that lead carries."""
    p = np.full(MAX_LEAD_HOURS + 1, LADDER_LONG_WEIGHT / (MAX_LEAD_HOURS + 1))
    p[: SHORT_MAX_LEAD + 1] += LADDER_SHORT_WEIGHT / (SHORT_MAX_LEAD + 1)
    return p / p.sum()


def parse_cycle(key: str) -> datetime:
    return datetime.strptime(key, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


@lru_cache(maxsize=24)
def _load_npy(path: str) -> np.ndarray:
    return np.load(path)


class EuropeCropDataset(Dataset):
    """One item = one (cycle, lead) sample on the domain window.

    Anti-overfit machinery (training only):
      - window jitter: the Germany 80x96 window shifts randomly inside the
        Europe crop each sample; statics, metric weights, coords and
        climatology shift with it, so the model cannot memorize pixels.
      - ladder-weighted lead sampling: leads <= 48h carry ~2.8x the incentive
        weight of longer leads (they score in both ladders) and are sampled
        accordingly.
      - lagged dropout: the lagged forecast pair is randomly replaced by the
        interpolation fallback, matching the serving path when the paired
        run is missing.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        cycles: list[str],
        source: str,
        statistics: dict[str, list[float]],
        crop_size: tuple[int, int] | None = None,
        samples_per_epoch: int | None = None,
        seed: int = 0,
        domain: str = "europe",
        loss_region: str = "europe",
        jitter: int = 0,
        lagged_dropout: float = 0.0,
        lead_weighting: str = "uniform",
    ) -> None:
        self.root = Path(root)
        self.cycles = list(cycles)
        self.source = source  # "single" or "ens"
        self.crop_size = crop_size
        self.domain = domain
        self.lagged_dropout = float(lagged_dropout)
        self.lead_probabilities = (
            lead_probabilities() if lead_weighting == "ladder" else None
        )
        self.rng = np.random.default_rng(seed)

        static = np.load(self.root / "static.npz")
        full_lat = static["latitudes"]
        full_lon = static["longitudes"]
        full_height, full_width = static["land"].shape

        if domain == "germany":
            base_lat, base_lon = GERMANY_LAT_SLICE, GERMANY_LON_SLICE
        elif domain == "europe":
            base_lat, base_lon = slice(0, full_height), slice(0, full_width)
        else:
            raise ValueError(f"Unknown domain {domain!r}")
        self._base_window = (base_lat, base_lon)
        self.height = base_lat.stop - base_lat.start
        self.width = base_lon.stop - base_lon.start
        # Jitter must keep the shifted window inside the Europe crop.
        self.jitter = min(
            int(jitter),
            base_lat.start,
            full_height - base_lat.stop,
            base_lon.start,
            full_width - base_lon.stop,
        )
        self.jitter = max(self.jitter, 0)

        def z(field: np.ndarray) -> np.ndarray:
            """z-score with statistics from the base window, applied globally."""
            ref = field[base_lat, base_lon]
            return (field - ref.mean()) / max(float(ref.std()), 1e-6)

        self._static_full = torch.from_numpy(
            np.stack(
                [
                    static["land"],
                    z(static["orography"]),
                    z(static["roughness"]),
                    static["cosine_latitude"],
                    z(np.log1p(static["temp_scalar"])),
                    z(np.log1p(static["wind_scalar"])),
                ]
            ).astype(np.float32)
        )

        # Official metric: per-variable cos-lat x capacity scalar, optionally
        # masked to Germany, normalized by its mean over the *base* window so
        # the loss scale does not change under jitter.
        germany_mask_full = (
            (full_lat[:, None] >= GERMANY_LAT[0])
            & (full_lat[:, None] <= GERMANY_LAT[1])
            & (full_lon[None, :] >= GERMANY_LON[0])
            & (full_lon[None, :] <= GERMANY_LON[1])
        ).astype(np.float32)
        cos_lat = static["cosine_latitude"]
        temp_metric = cos_lat * static["temp_scalar"]
        wind_metric = cos_lat * static["wind_scalar"]
        metric = np.stack([temp_metric, wind_metric, wind_metric])
        if loss_region == "germany":
            metric = metric * germany_mask_full[None]
        base_mean = metric[:, base_lat, base_lon].mean(axis=(1, 2), keepdims=True)
        metric = metric / np.maximum(base_mean, 1e-12)
        self._metric_full = torch.from_numpy(metric.astype(np.float32))
        self._clim_full = static["climatology_coefficients"]
        self._lat_full = torch.from_numpy(full_lat.copy())
        self._lon_full = torch.from_numpy(full_lon.copy())

        # Base-window views used by callers (mask counts, serving export).
        self.metric_weights = self._metric_full[:, base_lat, base_lon]
        self.germany_mask = torch.from_numpy(
            germany_mask_full[base_lat, base_lon].copy()
        )
        self.latitudes = self._lat_full[base_lat]
        self.longitudes = self._lon_full[base_lon]

        self.state_mean = np.asarray(statistics["state_mean"], np.float32)[
            :, None, None
        ]
        self.state_std = np.asarray(statistics["state_std"], np.float32)[
            :, None, None
        ]
        self.delta_std = np.asarray(statistics["delta_std"], np.float32)[
            :, None, None
        ]
        self.residual_std = np.asarray(statistics["residual_std"], np.float32)

        self.entries = [
            (cycle, lead)
            for cycle in range(len(self.cycles))
            for lead in range(MAX_LEAD_HOURS + 1)
        ]
        self.samples_per_epoch = samples_per_epoch or len(self.entries)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_window(self) -> tuple[slice, slice]:
        base_lat, base_lon = self._base_window
        if self.jitter == 0:
            return base_lat, base_lon
        dy = int(self.rng.integers(-self.jitter, self.jitter + 1))
        dx = int(self.rng.integers(-self.jitter, self.jitter + 1))
        return (
            slice(base_lat.start + dy, base_lat.stop + dy),
            slice(base_lon.start + dx, base_lon.stop + dx),
        )

    def _forecast(
        self, cycle_key: str, source: str, window: tuple[slice, slice]
    ) -> np.ndarray | None:
        path = self.root / source / f"{cycle_key}.npy"
        if not path.is_file():
            return None
        return _load_npy(str(path))[..., window[0], window[1]]

    def _truth(
        self, valid: datetime, window: tuple[slice, slice]
    ) -> np.ndarray | None:
        path = self.root / "era5" / f"{valid.strftime('%Y-%m-%d')}.npy"
        if not path.is_file():
            return None
        return (
            _load_npy(str(path))[valid.hour, :, window[0], window[1]]
            .astype(np.float32)
        )

    def _interpolate(self, steps: np.ndarray, lead: int) -> np.ndarray:
        left, right, fraction = bracket_for_lead(lead)
        return (
            (1.0 - fraction) * steps[left].astype(np.float32)
            + fraction * steps[right].astype(np.float32)
        )

    def _lagged(
        self,
        cycle_key: str,
        cycle_time: datetime,
        lead: int,
        window: tuple[slice, slice],
    ) -> np.ndarray | None:
        if self.source == "ens":
            # Serving pairs the ENS mean with the same-day AIFS Single run.
            steps = self._forecast(cycle_key, "single", window)
            return None if steps is None else self._interpolate(steps, lead)
        prev_key = (cycle_time - timedelta(days=1)).strftime("%Y%m%dT%H%M%SZ")
        if lead + 24 > MAX_LEAD_HOURS:
            return None
        steps = self._forecast(prev_key, "single", window)
        return None if steps is None else self._interpolate(steps, lead + 24)

    def sample_entry(self) -> tuple[int, int]:
        cycle = int(self.rng.integers(0, len(self.cycles)))
        if self.lead_probabilities is not None:
            lead = int(
                self.rng.choice(MAX_LEAD_HOURS + 1, p=self.lead_probabilities)
            )
        else:
            lead = int(self.rng.integers(0, MAX_LEAD_HOURS + 1))
        return cycle, lead

    def build_item(
        self,
        cycle_index: int,
        lead: int,
        crop: tuple[int, int] | None,
        augment: bool = False,
    ) -> dict[str, torch.Tensor] | None:
        cycle_key = self.cycles[cycle_index]
        cycle_time = parse_cycle(cycle_key)
        valid = cycle_time + timedelta(hours=lead)
        window = self._sample_window() if augment else self._base_window
        steps = self._forecast(cycle_key, self.source, window)
        truth = self._truth(valid, window)
        if steps is None or truth is None:
            return None
        interpolated = self._interpolate(steps, lead)
        left, right, _ = bracket_for_lead(lead)
        delta = steps[right].astype(np.float32) - steps[left].astype(np.float32)
        lagged = self._lagged(cycle_key, cycle_time, lead, window)
        if augment and self.lagged_dropout > 0.0:
            if self.rng.random() < self.lagged_dropout:
                lagged = None  # train the serving fallback path
        if lagged is None:
            lagged = interpolated
        clim = evaluate_climatology(
            self._clim_full[:, :, window[0], window[1]], valid
        )
        latitudes = self._lat_full[window[0]]
        longitudes = self._lon_full[window[1]]
        zenith_now = cosine_solar_zenith(latitudes, longitudes, valid)
        zenith_prev = cosine_solar_zenith(
            latitudes, longitudes, valid - timedelta(hours=2)
        )
        features = torch.from_numpy(
            np.concatenate(
                [
                    (interpolated - self.state_mean) / self.state_std,
                    delta / self.delta_std,
                    (lagged - self.state_mean) / self.state_std,
                    (lagged - interpolated) / self.delta_std,
                    (interpolated - clim) / self.state_std,
                ]
            ).astype(np.float32)
        )
        features = torch.cat(
            [
                features,
                zenith_now.clamp_min(0.0).unsqueeze(0),
                (zenith_now - zenith_prev).unsqueeze(0),
                self._static_full[:, window[0], window[1]],
            ]
        )
        item = {
            "features": features,
            "context": build_context(cycle_time, lead),
            "raw": torch.from_numpy(interpolated),
            "truth": torch.from_numpy(truth),
            "metric_weights": self._metric_full[:, window[0], window[1]],
        }
        if crop is not None:
            ch, cw = crop
            top = int(self.rng.integers(0, self.height - ch + 1))
            left_px = int(self.rng.integers(0, self.width - cw + 1))
            for key in ("features", "raw", "truth", "metric_weights"):
                item[key] = item[key][..., top : top + ch, left_px : left_px + cw]
        return item

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        for _ in range(20):
            cycle, lead = self.sample_entry()
            item = self.build_item(cycle, lead, self.crop_size, augment=True)
            if item is not None:
                return item
        raise RuntimeError("Could not sample a valid (cycle, lead) in 20 tries.")


def build_split(cycles: list[str], drop_after: str) -> tuple[list[str], list[str]]:
    limit = datetime.fromisoformat(drop_after).replace(tzinfo=timezone.utc)
    blocks = [
        (
            datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
            datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
        )
        for start, end in VALIDATION_BLOCKS
    ]
    train, validation = [], []
    for key in cycles:
        date = parse_cycle(key)
        if date > limit:
            continue
        if any(start <= date <= end for start, end in blocks):
            validation.append(key)
        elif not any(
            start - timedelta(days=BUFFER_DAYS)
            <= date
            <= end + timedelta(days=BUFFER_DAYS)
            for start, end in blocks
        ):
            train.append(key)
    return train, validation


def validation_entries(
    n_cycles: int, per_cycle: int, seed: int = 5
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    entries = []
    for cycle in range(n_cycles):
        edges = np.linspace(0, MAX_LEAD_HOURS, per_cycle + 1)
        for i in range(per_cycle):
            lead = int(rng.integers(int(edges[i]), int(edges[i + 1]) + 1))
            entries.append((cycle, min(lead, MAX_LEAD_HOURS)))
    return entries


def combined_error(
    prediction: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    error = prediction - truth
    rmse = torch.sqrt((error.square() * weights).mean(dim=(-2, -1)))
    mae = (error.abs() * weights).mean(dim=(-2, -1))
    return (rmse + mae) / 2.0


class EmaWeights:
    """Exponential moving average of model weights with bias-corrected warmup."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.updates = 0
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                self.shadow[key].copy_(value)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow)


@torch.no_grad()
def evaluate(
    model: EuropeResUNet,
    dataset: EuropeCropDataset,
    entries: list[tuple[int, int]],
    residual_scales: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    corrected_sum = np.zeros(3)
    raw_sum = np.zeros(3)
    used = 0
    for cycle, lead in entries:
        item = dataset.build_item(cycle, lead, None)
        if item is None:
            continue
        features = item["features"].unsqueeze(0).to(device)
        context = item["context"].unsqueeze(0).to(device)
        raw = item["raw"].unsqueeze(0).to(device)
        truth = item["truth"].unsqueeze(0).to(device)
        output = model(features, context)
        corrected = raw + output.gate * output.correction * residual_scales
        weights = item["metric_weights"].unsqueeze(0).to(device)
        corrected_sum += combined_error(corrected, truth, weights)[0].cpu().numpy()
        raw_sum += combined_error(raw, truth, weights)[0].cpu().numpy()
        used += 1
    if was_training:
        model.train()
    corrected_mean = corrected_sum / max(used, 1)
    raw_mean = raw_sum / max(used, 1)
    weights_np = np.asarray(VARIABLE_WEIGHTS)
    metrics: dict[str, float] = {"val_samples": float(used)}
    for i, key in enumerate(SHORT_NAMES):
        metrics[f"{key}_corrected"] = float(corrected_mean[i])
        metrics[f"{key}_baseline"] = float(raw_mean[i])
        metrics[f"{key}_gain_pct"] = float(
            100.0 * (raw_mean[i] - corrected_mean[i]) / max(raw_mean[i], 1e-9)
        )
    metrics["weighted_corrected"] = float(
        (corrected_mean / np.maximum(raw_mean, 1e-9) * weights_np).sum()
    )
    metrics["weighted_gain_pct"] = float(
        100.0 * (1.0 - metrics["weighted_corrected"] / weights_np.sum())
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Zeus/data/evaluation/europe_crops")
    parser.add_argument("--output-root", default="/Zeus/data/evaluation/training")
    parser.add_argument("--name", default="europe_resunet_v1")
    parser.add_argument("--source", choices=("single", "ens"), default="single")
    parser.add_argument(
        "--domain",
        choices=("europe", "germany"),
        default="europe",
        help="europe: 208x368 crop. germany: 80x96 slice around DE with context.",
    )
    parser.add_argument("--loss-region", choices=("europe", "germany"), default="europe")
    parser.add_argument(
        "--statistics",
        default="/Zeus/data/evaluation/training/aifs_downscaler_v2_ens.statistics.json",
    )
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--drop-after", default="2026-07-11")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-height", type=int, default=176)
    parser.add_argument("--crop-width", type=int, default=320)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-regret-weight", type=float, default=0.5)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout2d inside residual blocks. Use ~0.10 for the small ENS "
        "fine-tune, 0 for pretraining.",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=8,
        help="Max window shift in cells (Germany domain only). The 80x96 "
        "window moves +-jitter inside the Europe crop each training sample.",
    )
    parser.add_argument(
        "--lagged-dropout",
        type=float,
        default=0.1,
        help="Probability of replacing the lagged forecast with the "
        "interpolation fallback during training (matches serving fallback).",
    )
    parser.add_argument(
        "--lead-weighting",
        choices=("ladder", "uniform"),
        default="ladder",
        help="ladder: sample leads proportional to Zeus incentive mass "
        "(leads <=48h ~2.8x). uniform: previous behavior.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Stop after this many epochs without validation improvement.",
    )
    parser.add_argument("--validation-per-cycle", type=int, default=6)
    parser.add_argument("--max-validation-cycles", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = device.type == "cuda"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / f"{args.name}.pt"
    metrics_path = output_root / f"{args.name}.metrics.jsonl"

    statistics = json.loads(Path(args.statistics).read_text(encoding="utf-8"))
    manifest = json.loads(
        (Path(args.data_root) / "manifest.json").read_text(encoding="utf-8")
    )
    all_cycles = manifest[f"{args.source}_cycles"]
    train_cycles, val_cycles = build_split(all_cycles, args.drop_after)
    print(
        f"{args.source}: {len(all_cycles)} cycles -> {len(train_cycles)} train, "
        f"{len(val_cycles)} validation",
        flush=True,
    )

    if args.domain == "germany":
        # Full 80x96 window fits a 4090 batch; random 176x320 crops do not.
        if args.crop_height >= GERMANY_HEIGHT or args.crop_width >= GERMANY_WIDTH:
            crop_size = None
        else:
            crop_size = (args.crop_height, args.crop_width)
        if args.loss_region == "europe":
            args.loss_region = "germany"
        model_cls = GermanyResUNet
        print(
            f"Germany domain {GERMANY_HEIGHT}x{GERMANY_WIDTH} "
            f"(42-61.75N, 2W-21.75E), loss={args.loss_region}, "
            f"jitter=+-{args.jitter} cells",
            flush=True,
        )
    else:
        crop_size = (args.crop_height, args.crop_width)
        model_cls = EuropeResUNet
    common = dict(
        root=args.data_root,
        source=args.source,
        statistics=statistics,
        domain=args.domain,
        loss_region=args.loss_region,
    )
    train_dataset = EuropeCropDataset(
        cycles=train_cycles,
        crop_size=crop_size,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        jitter=args.jitter if args.domain == "germany" else 0,
        lagged_dropout=args.lagged_dropout,
        lead_weighting=args.lead_weighting,
        **common,
    )
    val_dataset = EuropeCropDataset(cycles=val_cycles, seed=args.seed + 1, **common)
    val_entries = validation_entries(
        min(len(val_cycles), args.max_validation_cycles), args.validation_per_cycle
    )
    print(
        f"domain grid {train_dataset.height}x{train_dataset.width} "
        f"germany cells in mask={int(train_dataset.germany_mask.sum())} "
        f"jitter={train_dataset.jitter} lead_weighting={args.lead_weighting} "
        f"lagged_dropout={args.lagged_dropout}",
        flush=True,
    )

    model = model_cls(
        base_channels=args.base_channels,
        blocks_per_stage=args.blocks_per_stage,
        in_channels=IN_CHANNELS,
        dropout=args.dropout,
    ).to(device)
    if args.init_from:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(init["model_state"])
        print(f"warm-started from {args.init_from}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M parameters on {device}", flush=True)

    criterion = ValidatorAwareResidualLoss(
        variable_weights=VARIABLE_WEIGHTS,
        solar_channel=None,
        no_regret_weight=args.no_regret_weight,
    )
    residual_scales = (
        torch.tensor(train_dataset.residual_std, dtype=torch.float32)
        .view(1, 3, 1, 1)
        .to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(args.samples_per_epoch / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=0.15,
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    baseline = evaluate(model, val_dataset, val_entries, residual_scales, device)
    print(
        "baseline (gate closed) "
        + " ".join(f"{k}={baseline[f'{k}_baseline']:.4f}" for k in SHORT_NAMES),
        flush=True,
    )

    ema = EmaWeights(model, args.ema_decay)
    ema_model = model_cls(
        base_channels=args.base_channels,
        blocks_per_stage=args.blocks_per_stage,
        in_channels=IN_CHANNELS,
        dropout=args.dropout,
    ).to(device)
    ema_model.eval()

    best = math.inf
    best_epoch = -1
    with metrics_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "baseline", "metrics": baseline}) + "\n")
        for epoch in range(args.epochs):
            t_epoch = time.time()
            running_loss = 0.0
            for step, item in enumerate(loader, start=1):
                features = item["features"].to(device, non_blocking=True)
                context = item["context"].to(device, non_blocking=True)
                raw = item["raw"].to(device, non_blocking=True)
                truth = item["truth"].to(device, non_blocking=True)
                weights = item["metric_weights"].to(device, non_blocking=True)
                with torch.autocast(device.type, enabled=use_amp):
                    output = model(features, context)
                    result = criterion(
                        raw_forecast=raw,
                        truth=truth,
                        correction=output.gate * output.correction,
                        gate=output.gate,
                        metric_weights=weights,
                        residual_scales=residual_scales,
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(result.loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                ema.update(model)
                running_loss += float(result.loss.detach())
                if step % 50 == 0:
                    print(
                        f"  epoch {epoch} step {step}/{steps_per_epoch} "
                        f"loss={running_loss / step:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"({(time.time() - t_epoch) / step:.2f}s/step)",
                        flush=True,
                    )
            raw_metrics = evaluate(
                model, val_dataset, val_entries, residual_scales, device
            )
            ema.copy_to(ema_model)
            ema_metrics = evaluate(
                ema_model, val_dataset, val_entries, residual_scales, device
            )
            use_ema = (
                ema_metrics["weighted_corrected"]
                <= raw_metrics["weighted_corrected"]
            )
            metrics = dict(ema_metrics if use_ema else raw_metrics)
            metrics["epoch"] = epoch
            metrics["selected"] = "ema" if use_ema else "raw"
            metrics["raw_weighted"] = raw_metrics["weighted_corrected"]
            metrics["ema_weighted"] = ema_metrics["weighted_corrected"]
            metrics["train_loss"] = running_loss / max(steps_per_epoch, 1)
            metrics["seconds"] = time.time() - t_epoch
            log.write(json.dumps({"event": "epoch", "metrics": metrics}) + "\n")
            log.flush()
            print(
                f"epoch {epoch}: loss={metrics['train_loss']:.4f} "
                f"gain t2m={metrics['t2m_gain_pct']:+.2f}% "
                f"u100={metrics['u100_gain_pct']:+.2f}% "
                f"v100={metrics['v100_gain_pct']:+.2f}% "
                f"weighted={metrics['weighted_gain_pct']:+.2f}% "
                f"[{metrics['selected']}] "
                f"({metrics['seconds'] / 60:.1f} min)",
                flush=True,
            )
            score = metrics["weighted_corrected"]
            if score < best:
                best = score
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": (
                            ema.shadow if use_ema else model.state_dict()
                        ),
                        "base_channels": args.base_channels,
                        "blocks_per_stage": args.blocks_per_stage,
                        "in_channels": IN_CHANNELS,
                        "dropout": args.dropout,
                        "source": args.source,
                        "domain": args.domain,
                        "loss_region": args.loss_region,
                        "statistics": statistics,
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                    checkpoint_path,
                )
                print(f"  saved checkpoint (weighted={score:.5f})", flush=True)
            elif args.patience > 0 and epoch - best_epoch >= args.patience:
                print(
                    f"early stop: no improvement for {args.patience} epochs "
                    f"(best epoch {best_epoch})",
                    flush=True,
                )
                break
    print(f"done. best weighted {best:.5f} -> {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
