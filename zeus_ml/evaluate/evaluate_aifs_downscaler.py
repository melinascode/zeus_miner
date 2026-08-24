"""Score the AIFS hourly downscaler with Zeus 48h / 360h iwRMSE and iwMAE."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from evaluation.scoring import ValidatorFaithfulScorer
from zeus.utils.coordinates import get_grid
from zeus.utils.region_mask import (
    OLD_REGION_CONFIGS,
    REGION_CONFIGS,
    build_geographic_weights,
)
from zeus_ml.datasets.aifs_downscale_dataset import (
    AifsCycleReader,
    EnsMeanCycleReader,
    Era5HourlyReader,
)
from zeus_ml.models.aifs_downscaler_cnn import (
    MAX_LEAD_HOURS,
    AifsDownscalerCNN,
    DownscalerStatistics,
    bracket_for_lead,
    build_downscaler_context,
    build_downscaler_static_features,
    load_static_maps,
    zenith_triplet,
)
from zeus_ml.train.train_aifs_downscaler import VALIDATION_BLOCKS, build_split


SHORT_NAMES = ("2t", "100u", "100v")
DEFAULT_CYCLES = (
    "20250721T000000Z",
    "20251021T000000Z",
    "20260120T000000Z",
    "20260414T000000Z",
)


class Stream:
    def __init__(self, n_vars: int = 3) -> None:
        self.n_vars = n_vars
        self.squared = torch.zeros(n_vars, dtype=torch.float64)
        self.absolute = torch.zeros(n_vars, dtype=torch.float64)
        self.cells = 0

    def update(self, prediction: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor) -> None:
        error = prediction - truth
        w = weights.unsqueeze(0)
        self.squared += (error.square() * w).sum(dim=(-2, -1)).double()
        self.absolute += (error.abs() * w).sum(dim=(-2, -1)).double()
        self.cells += error.shape[-2] * error.shape[-1]

    def finalize(self) -> list[dict[str, float]]:
        mse = self.squared / max(self.cells, 1)
        mae = self.absolute / max(self.cells, 1)
        rmse = torch.sqrt(mse)
        return [
            {
                "rmse": float(rmse[i]),
                "mae": float(mae[i]),
                "combined": float((rmse[i] + mae[i]) / 2.0),
            }
            for i in range(self.n_vars)
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/Zeus/data/evaluation/training/aifs_downscaler_v1.pt",
    )
    parser.add_argument("--aifs-root", default="/Zeus/data/evaluation/aifs_single")
    parser.add_argument(
        "--ens-root",
        default=None,
        help="Score ENS-mean cycles; lagged slot carries the same-day Single run.",
    )
    parser.add_argument("--era5-root", default="/Zeus/data/evaluation/era5")
    parser.add_argument("--static-root", default="/Zeus/data/evaluation/training/aifs_static")
    parser.add_argument("--cycle", action="append", default=[])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-lead", type=int, default=MAX_LEAD_HOURS)
    parser.add_argument(
        "--output",
        default="/Zeus/data/evaluation/training/aifs_downscaler_v1.horizon_scores.json",
    )
    return parser.parse_args()


def cycle_maps(
    cycle_time: datetime, latitudes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = get_grid(-90.0, 90.0, -180.0, 179.75)
    regime = ValidatorFaithfulScorer.region_regime(cycle_time)
    configs = OLD_REGION_CONFIGS if regime == "europe_only" else REGION_CONFIGS
    geographic = build_geographic_weights(grid, configs)
    metric = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)[:, None] * geographic
    return geographic, metric / metric.mean()


def evaluate_cycle(
    *,
    cycle_key: str,
    model: AifsDownscalerCNN,
    statistics: DownscalerStatistics,
    reader: AifsCycleReader | EnsMeanCycleReader,
    truth_reader: Era5HourlyReader,
    single_reader: AifsCycleReader | None = None,
    land: torch.Tensor,
    orography: torch.Tensor,
    roughness: torch.Tensor,
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    max_lead: int = MAX_LEAD_HOURS,
    use_lagged: bool = False,
) -> dict:
    cycle_time = datetime.strptime(cycle_key, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    geographic, weights = cycle_maps(cycle_time, latitudes)
    mean, std, delta_std, residual_std = statistics.tensors()
    aifs = reader.get(cycle_key)
    horizons = [h for h in (48, 360) if h <= max_lead]
    if max_lead not in horizons:
        horizons.append(max_lead)
    streams = {h: {"raw": Stream(), "cnn": Stream()} for h in horizons}
    t0 = time.time()
    with torch.inference_mode():
        for lead in range(max_lead + 1):
            left, right, fraction = bracket_for_lead(lead)
            a_left = torch.from_numpy(aifs[left].astype(np.float32))
            a_right = torch.from_numpy(aifs[right].astype(np.float32))
            interpolated = (1.0 - fraction) * a_left + fraction * a_right
            delta = a_right - a_left
            blocks = [(interpolated - mean) / std, delta / delta_std]
            if use_lagged:
                pair_reader = single_reader if single_reader is not None else reader
                if single_reader is not None:
                    # ENS mode: the paired forecast is the same-day Single run.
                    prev_key = cycle_key
                    lag_lead = lead
                else:
                    prev_key = (cycle_time - timedelta(days=1)).strftime(
                        "%Y%m%dT%H%M%SZ"
                    )
                    lag_lead = lead + 24
                if (
                    pair_reader.path_for(prev_key).is_file()
                    and lag_lead <= MAX_LEAD_HOURS
                ):
                    prev = pair_reader.get(prev_key)
                    l2, r2, f2 = bracket_for_lead(lag_lead)
                    lagged = (1.0 - f2) * torch.from_numpy(
                        prev[l2].astype(np.float32)
                    ) + f2 * torch.from_numpy(prev[r2].astype(np.float32))
                else:
                    lagged = interpolated
                blocks += [(lagged - mean) / std, (lagged - interpolated) / delta_std]
            weather = torch.cat(blocks, dim=0).unsqueeze(0)
            zenith, zenith_anomaly = zenith_triplet(
                latitudes, longitudes, cycle_time, lead
            )
            static = build_downscaler_static_features(
                latitudes,
                longitudes,
                geographic_weights=geographic,
                land_sea=land,
                orography=orography,
                roughness=roughness,
                zenith=zenith,
                zenith_anomaly=zenith_anomaly,
            ).unsqueeze(0)
            context = build_downscaler_context(
                lead_hour=lead,
                cycle_hour=cycle_time.hour,
                day_of_year=cycle_time.timetuple().tm_yday,
                fraction=fraction,
            ).unsqueeze(0)
            output = model(weather, context, static)
            corrected = interpolated + output.correction[0] * residual_std
            truth = torch.from_numpy(
                truth_reader.read(cycle_time + timedelta(hours=lead))
            )
            for horizon, pair in streams.items():
                if lead <= horizon:
                    pair["raw"].update(interpolated, truth, weights)
                    pair["cnn"].update(corrected, truth, weights)
            if lead in (48, 360) or lead % 60 == 0:
                print(
                    f"  {cycle_key} lead {lead:3d}/{max_lead} "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
    return {
        "cycle": cycle_key,
        "region_regime": ValidatorFaithfulScorer.region_regime(cycle_time),
        "seconds": time.time() - t0,
        "horizons": {
            str(h): {
                "linear": {n: s["raw"].finalize()[i] for i, n in enumerate(SHORT_NAMES)},
                "downscaler": {
                    n: s["cnn"].finalize()[i] for i, n in enumerate(SHORT_NAMES)
                },
            }
            for h, s in streams.items()
        },
    }


def mean_over_cycles(results: list[dict], horizon: str, source: str) -> dict:
    out = {}
    for name in SHORT_NAMES:
        rmses = [r["horizons"][horizon][source][name]["rmse"] for r in results]
        maes = [r["horizons"][horizon][source][name]["mae"] for r in results]
        out[name] = {
            "rmse": float(np.mean(rmses)),
            "mae": float(np.mean(maes)),
        }
    return out


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    statistics = DownscalerStatistics.from_dict(checkpoint["statistics"])
    use_lagged = bool(checkpoint.get("use_lagged", False))
    weather_channels = int(checkpoint.get("weather_channels", 12 if use_lagged else 6))
    model = AifsDownscalerCNN(
        hidden_channels=checkpoint["hidden_channels"],
        weather_channels=weather_channels,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(
        f"checkpoint epoch={checkpoint.get('epoch')} "
        f"hidden={checkpoint['hidden_channels']} "
        f"weather_channels={weather_channels} lagged={use_lagged}",
        flush=True,
    )

    ens_mode = bool(args.ens_root) or bool(checkpoint.get("ens_mode", False))
    if ens_mode and not args.ens_root:
        raise SystemExit("Checkpoint was trained on ENS means; pass --ens-root.")
    if ens_mode:
        available = {p.stem for p in Path(args.ens_root).glob("*.npy")}
    else:
        available = {p.stem for p in Path(args.aifs_root).glob("*.grib2")}
    if args.cycle:
        cycles = args.cycle
    else:
        cycles = [c for c in DEFAULT_CYCLES if c in available]
        if len(cycles) < 4:
            split = build_split(sorted(available))
            # One cycle from the middle of each seasonal holdout block.
            cycles = []
            for start, end in VALIDATION_BLOCKS:
                lo = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
                hi = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
                mid = [
                    k
                    for k in split.validation
                    if lo <= datetime.strptime(k, "%Y%m%dT%H%M%SZ").replace(
                        tzinfo=timezone.utc
                    )
                    <= hi
                ]
                if mid:
                    cycles.append(mid[len(mid) // 2])
    missing = [c for c in cycles if c not in available]
    if missing:
        raise SystemExit(f"Missing forecast files: {missing}")

    land, orography, roughness = load_static_maps(args.static_root)
    latitudes = torch.linspace(-90.0, 90.0, 721)
    longitudes = torch.arange(-180.0, 180.0, 0.25)
    single_reader = None
    if ens_mode:
        reader = EnsMeanCycleReader(args.ens_root, cache_size=1)
        if use_lagged:
            single_reader = AifsCycleReader(args.aifs_root, cache_size=1)
    else:
        reader = AifsCycleReader(args.aifs_root, cache_size=3 if use_lagged else 1)
    truth_reader = Era5HourlyReader(args.era5_root, cache_size=9)

    print(f"scoring {len(cycles)} cycles: {cycles}", flush=True)
    results = []
    for cycle_key in cycles:
        print(f"cycle {cycle_key}", flush=True)
        result = evaluate_cycle(
            cycle_key=cycle_key,
            model=model,
            statistics=statistics,
            reader=reader,
            truth_reader=truth_reader,
            single_reader=single_reader,
            land=land,
            orography=orography,
            roughness=roughness,
            latitudes=latitudes,
            longitudes=longitudes,
            max_lead=args.max_lead,
            use_lagged=use_lagged,
        )
        results.append(result)
        for horizon in sorted(result["horizons"], key=int):
            lin = result["horizons"][horizon]["linear"]
            cnn = result["horizons"][horizon]["downscaler"]
            print(
                f"  {horizon}h  linear "
                + "  ".join(
                    f"{n}={lin[n]['rmse']:.3f}/{lin[n]['mae']:.3f}" for n in SHORT_NAMES
                )
                + "  | cnn "
                + "  ".join(
                    f"{n}={cnn[n]['rmse']:.3f}/{cnn[n]['mae']:.3f}" for n in SHORT_NAMES
                ),
                flush=True,
            )

    horizon_keys = sorted(results[0]["horizons"], key=int)
    summary = {
        f"{h}h": {
            "linear": mean_over_cycles(results, h, "linear"),
            "downscaler": mean_over_cycles(results, h, "downscaler"),
        }
        for h in horizon_keys
    }
    payload = {
        "checkpoint": args.checkpoint,
        "epoch": checkpoint.get("epoch"),
        "cycles": results,
        "summary": summary,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "summary": summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
