#!/usr/bin/env python3
"""Decompose ENS-mean ± downscaler error over Germany / Europe / world boxes.

For 48h and 360h:

  current          today's validator metric (cosine-lat × 1.5 Europe / 2.5 Germany)
  germany
  rest_europe
  europe
  rest_of_world    cosine-lat skill inside that box
  mass_40_40_20    Tuesday box-only proxy: 40% DE / 40% rest-EU / 20% rest-of-world
                   (no wind / solar / population maps yet)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from zeus.utils.coordinates import get_grid
from zeus.utils.region_mask import (
    REGION_CONFIGS,
    build_geographic_weights,
    build_mass_share_weights,
    geographic_scalar_for_variable,
    region_masks_for_grid,
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

SHORT_NAMES = ("2t", "100u", "100v")
HORIZONS = (48, 360)
REGION_ORDER = (
    "current",
    "official_temp",
    "official_wind",
    "mass_40_40_20",
    "germany",
    "rest_europe",
    "europe",
    "rest_of_world",
)
# Live-dashboard UID 141, current 1.5/2.5 metric. Regional leader numbers unknown.
LEADER_CURRENT = {
    48: {
        "2t": {"rmse": 0.725, "mae": 0.470},
        "100u": {"rmse": 1.097, "mae": 0.747},
        "100v": {"rmse": 1.155, "mae": 0.777},
    },
    360: {
        "2t": {"rmse": 1.549, "mae": 0.926},
        "100u": {"rmse": 3.216, "mae": 2.093},
        "100v": {"rmse": 3.506, "mae": 2.251},
    },
}


class MassStream:
    """Weighted RMSE/MAE via sum(w·err) / sum(w). ``w`` is unnormalized."""

    def __init__(self, n_vars: int = 3) -> None:
        self.squared = torch.zeros(n_vars, dtype=torch.float64)
        self.absolute = torch.zeros(n_vars, dtype=torch.float64)
        self.mass = 0.0

    def update(
        self,
        prediction: torch.Tensor,
        truth: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        error = prediction - truth
        w = weights.unsqueeze(0)
        self.squared += (error.square() * w).sum(dim=(-2, -1)).double()
        self.absolute += (error.abs() * w).sum(dim=(-2, -1)).double()
        self.mass += float(weights.sum())

    def finalize(self) -> list[dict[str, float]]:
        denom = max(self.mass, 1e-18)
        mse = self.squared / denom
        mae = self.absolute / denom
        rmse = torch.sqrt(mse)
        return [
            {
                "rmse": float(rmse[i]),
                "mae": float(mae[i]),
                "combined": float((rmse[i] + mae[i]) / 2.0),
            }
            for i in range(len(SHORT_NAMES))
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/Zeus/data/evaluation/training/aifs_downscaler_v2_ens.pt",
    )
    parser.add_argument("--ens-root", default="/Zeus/data/evaluation/aifs_ens_mean")
    parser.add_argument("--aifs-root", default="/Zeus/data/evaluation/aifs_single")
    parser.add_argument("--era5-root", default="/Zeus/data/evaluation/era5")
    parser.add_argument(
        "--static-root", default="/Zeus/data/evaluation/training/aifs_static"
    )
    parser.add_argument("--cycle", action="append", default=[])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-lead", type=int, default=MAX_LEAD_HOURS)
    parser.add_argument(
        "--output",
        default="/Zeus/data/evaluation/training/regional_box_scores.json",
    )
    return parser.parse_args()


def named_weight_maps(
    latitudes: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, int], dict[str, float]]:
    grid = get_grid(-90.0, 90.0, -180.0, 179.75)
    cosine = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)
    cosine_2d = cosine[:, None]
    masks = region_masks_for_grid(grid)
    current_geo = build_geographic_weights(grid, REGION_CONFIGS)
    maps = {
        "current": cosine_2d * current_geo,
        "cosine_global": cosine_2d.expand(-1, current_geo.shape[1]).contiguous(),
        "germany": cosine_2d * masks["germany"],
        "rest_europe": cosine_2d * masks["rest_europe"],
        "europe": cosine_2d * masks["europe"],
        "rest_of_world": cosine_2d * masks["rest_of_world"],
        "mass_40_40_20": build_mass_share_weights(grid, cosine),
        # Official validator scalars active for challenges starting
        # 2026-08-25 18:00 UTC (temp map scores 2t, wind map scores 100u/100v).
        "official_temp": cosine_2d
        * geographic_scalar_for_variable("2m_temperature"),
        "official_wind": cosine_2d
        * geographic_scalar_for_variable("100m_u_component_of_wind"),
    }
    counts = {name: int((masks[name] > 0).sum()) for name in masks}
    mass = maps["mass_40_40_20"]
    shares = {
        key: float((mass * masks[key]).sum() / mass.sum())
        for key in ("germany", "rest_europe", "rest_of_world")
    }
    print(
        "box cells  "
        + " ".join(f"{k}={v}" for k, v in counts.items())
        + "  mass_40_40_20 shares  "
        + " ".join(f"{k}={s:.3f}" for k, s in shares.items()),
        flush=True,
    )
    return maps, counts, shares


def pack_streams(
    streams: dict[str, MassStream],
) -> dict[str, dict[str, dict[str, float]]]:
    packed: dict[str, dict[str, dict[str, float]]] = {}
    for region, stream in streams.items():
        scores = stream.finalize()
        packed[region] = {
            SHORT_NAMES[i]: scores[i] for i in range(len(SHORT_NAMES))
        }
    return packed


def evaluate_cycle(
    *,
    cycle_key: str,
    model: AifsDownscalerCNN,
    statistics: DownscalerStatistics,
    reader: EnsMeanCycleReader,
    truth_reader: Era5HourlyReader,
    single_reader: AifsCycleReader | None,
    land: torch.Tensor,
    orography: torch.Tensor,
    roughness: torch.Tensor,
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    weight_maps: dict[str, torch.Tensor],
    geo_feature: torch.Tensor,
    max_lead: int,
    use_lagged: bool,
) -> dict:
    cycle_time = datetime.strptime(cycle_key, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    mean, std, delta_std, residual_std = statistics.tensors()
    aifs = reader.get(cycle_key)
    horizons = [h for h in HORIZONS if h <= max_lead]
    streams = {
        h: {
            source: {name: MassStream() for name in weight_maps}
            for source in ("linear", "downscaler")
        }
        for h in horizons
    }
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
                if (
                    single_reader is not None
                    and single_reader.path_for(cycle_key).is_file()
                    and lead <= MAX_LEAD_HOURS
                ):
                    prev = single_reader.get(cycle_key)
                    l2, r2, f2 = bracket_for_lead(lead)
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
                geographic_weights=geo_feature,
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
            for horizon, by_source in streams.items():
                if lead > horizon:
                    continue
                for name, weights in weight_maps.items():
                    by_source["linear"][name].update(interpolated, truth, weights)
                    by_source["downscaler"][name].update(corrected, truth, weights)
            if lead in (48, 360) or lead % 60 == 0:
                print(
                    f"  {cycle_key} lead {lead:3d}/{max_lead} "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
    return {
        "cycle": cycle_key,
        "seconds": time.time() - t0,
        "horizons": {
            str(h): {
                source: pack_streams(by_region)
                for source, by_region in by_source.items()
            }
            for h, by_source in streams.items()
        },
    }


def mean_cycles(results: list[dict], horizon: str, source: str, region: str) -> dict:
    out = {}
    for name in SHORT_NAMES:
        rmses = [
            r["horizons"][horizon][source][region][name]["rmse"] for r in results
        ]
        maes = [r["horizons"][horizon][source][region][name]["mae"] for r in results]
        out[name] = {
            "rmse": float(np.mean(rmses)),
            "mae": float(np.mean(maes)),
            "combined": float((np.mean(rmses) + np.mean(maes)) / 2.0),
        }
    return out


def gap_vs_leader(ours: dict, horizon: int) -> dict:
    leader = LEADER_CURRENT[horizon]
    return {
        name: {
            "rmse_vs_leader_pct": 100.0
            * (ours[name]["rmse"] / leader[name]["rmse"] - 1.0),
            "mae_vs_leader_pct": 100.0
            * (ours[name]["mae"] / leader[name]["mae"] - 1.0),
        }
        for name in SHORT_NAMES
    }


def print_block(title: str, scores: dict) -> None:
    parts = [
        f"{n}={scores[n]['rmse']:.3f}/{scores[n]['mae']:.3f}" for n in SHORT_NAMES
    ]
    print(f"  {title:22s}  " + "  ".join(parts), flush=True)


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    cycles = args.cycle or ["20260727T000000Z", "20260801T000000Z"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    statistics = DownscalerStatistics.from_dict(checkpoint["statistics"])
    use_lagged = bool(checkpoint.get("use_lagged", False))
    weather_channels = int(
        checkpoint.get("weather_channels", 12 if use_lagged else 6)
    )
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

    land, orography, roughness = load_static_maps(args.static_root)
    latitudes = torch.linspace(-90.0, 90.0, 721)
    longitudes = torch.arange(-180.0, 180.0, 0.25)
    weight_maps, cell_counts, mass_shares = named_weight_maps(latitudes)
    grid = get_grid(-90.0, 90.0, -180.0, 179.75)
    geo_feature = build_geographic_weights(grid, REGION_CONFIGS)
    reader = EnsMeanCycleReader(args.ens_root, cache_size=1)
    single_reader = AifsCycleReader(args.aifs_root, cache_size=1)
    truth_reader = Era5HourlyReader(args.era5_root, cache_size=9)

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
            weight_maps=weight_maps,
            geo_feature=geo_feature,
            max_lead=args.max_lead,
            use_lagged=use_lagged,
        )
        results.append(result)
        for horizon in sorted(result["horizons"], key=int):
            print(f"  -- {horizon}h --", flush=True)
            for region in REGION_ORDER:
                print_block(
                    f"{region} linear",
                    result["horizons"][horizon]["linear"][region],
                )
                print_block(
                    f"{region} cnn",
                    result["horizons"][horizon]["downscaler"][region],
                )

    summary = {
        f"{h}h": {
            source: {
                region: mean_cycles(results, str(h), source, region)
                for region in weight_maps
            }
            for source in ("linear", "downscaler")
        }
        for h in HORIZONS
        if h <= args.max_lead
    }
    for h in HORIZONS:
        if h > args.max_lead:
            continue
        summary[f"{h}h"]["downscaler_current_vs_leader"] = gap_vs_leader(
            summary[f"{h}h"]["downscaler"]["current"], h
        )

    payload = {
        "checkpoint": args.checkpoint,
        "cycles": results,
        "cell_counts": cell_counts,
        "mass_40_40_20_shares": mass_shares,
        "leader_current_metric": LEADER_CURRENT,
        "note": (
            "germany/rest_europe/rest_of_world are cosine-lat skill inside the box. "
            "mass_40_40_20 is the Tuesday 40/40/20 box-only proxy. "
            "Leader numbers are on the *current* 1.5/2.5 metric, not regional."
        ),
        "summary": summary,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== mean over cycles ===", flush=True)
    for h in HORIZONS:
        if h > args.max_lead:
            continue
        print(f"{h}h downscaler", flush=True)
        for region in REGION_ORDER:
            print_block(region, summary[f"{h}h"]["downscaler"][region])
        gap = summary[f"{h}h"]["downscaler_current_vs_leader"]
        print(
            "  vs UID141 current  "
            + "  ".join(
                f"{n} {gap[n]['rmse_vs_leader_pct']:+.1f}% rmse" for n in SHORT_NAMES
            ),
            flush=True,
        )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
