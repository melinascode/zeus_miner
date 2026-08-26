#!/usr/bin/env python3
"""Stage 0b: lead-dependent climatology shrinkage under the official metric.

Fit α(lead, variable) on summer ENS-mean cycles that do not overlap the two
held-out test cycles, then score:

  climatology
  linear interpolation
  linear + shrinkage
  CNN downscaler
  CNN + shrinkage

on 20260727 / 20260801. α is the weighted-MSE-optimal blend of forecast and
climatology, using the official capacity scalars (temp map for 2t, wind map
for 100u/100v). Clipped to [0, 1] and smoothed over a 13-hour window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_era5_climatology import evaluate_climatology
from zeus.utils.coordinates import get_grid
from zeus.utils.region_mask import (
    REGION_CONFIGS,
    build_geographic_weights,
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
FIT_CYCLES = (
    "20260520T000000Z",
    "20260527T000000Z",
    "20260603T000000Z",
    "20260610T000000Z",
    "20260617T000000Z",
    "20260624T000000Z",
    "20260701T000000Z",
    "20260708T000000Z",
)
TEST_CYCLES = ("20260727T000000Z", "20260801T000000Z")
SMOOTH_WINDOW = 13


class MassStream:
    def __init__(self, n_vars: int = 3) -> None:
        self.squared = np.zeros(n_vars, dtype=np.float64)
        self.absolute = np.zeros(n_vars, dtype=np.float64)
        self.mass = np.zeros(n_vars, dtype=np.float64)

    def update(
        self,
        prediction: np.ndarray,
        truth: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        error = prediction - truth
        w = weights
        self.squared += (error * error * w).sum(axis=(-2, -1))
        self.absolute += (np.abs(error) * w).sum(axis=(-2, -1))
        self.mass += w.sum(axis=(-2, -1))

    def finalize(self) -> dict[str, dict[str, float]]:
        denom = np.maximum(self.mass, 1e-18)
        rmse = np.sqrt(self.squared / denom)
        mae = self.absolute / denom
        return {
            SHORT_NAMES[i]: {
                "rmse": float(rmse[i]),
                "mae": float(mae[i]),
                "combined": float((rmse[i] + mae[i]) / 2.0),
            }
            for i in range(len(SHORT_NAMES))
        }


def parse_cycle(key: str) -> datetime:
    return datetime.strptime(key, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def smooth_alpha(alpha: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Centered moving average; keep α[0]=0 and clip to [0, 1]."""
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    out = np.empty_like(alpha)
    for v in range(alpha.shape[1]):
        padded = np.pad(alpha[:, v], (pad, pad), mode="edge")
        out[:, v] = np.convolve(padded, kernel, mode="valid")[: alpha.shape[0]]
    out[0] = 0.0
    return np.clip(out, 0.0, 1.0)


def build_weight_maps(latitudes: torch.Tensor) -> dict[str, np.ndarray]:
    cosine = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)[:, None]
    grid = get_grid(-90.0, 90.0, -180.0, 179.75)
    masks = region_masks_for_grid(grid)
    temp = (cosine * geographic_scalar_for_variable("2m_temperature")).numpy()
    wind = (
        cosine * geographic_scalar_for_variable("100m_u_component_of_wind")
    ).numpy()
    official = np.stack([temp, wind, wind]).astype(np.float32)
    germany = (cosine * masks["germany"]).numpy().astype(np.float32)
    rest_europe = (cosine * masks["rest_europe"]).numpy().astype(np.float32)
    return {
        "official": official,
        "germany": np.stack([germany, germany, germany]),
        "rest_europe": np.stack([rest_europe, rest_europe, rest_europe]),
    }


def interpolate_cycle(aifs: np.ndarray, lead: int) -> np.ndarray:
    left, right, fraction = bracket_for_lead(lead)
    return (
        (1.0 - fraction) * aifs[left].astype(np.float32)
        + fraction * aifs[right].astype(np.float32)
    )


def blend(forecast: np.ndarray, clim: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha.astype(np.float32)[:, None, None]
    return (1.0 - a) * forecast + a * clim


def fit_alpha(
    *,
    cycles: tuple[str, ...],
    reader: EnsMeanCycleReader,
    truth_reader: Era5HourlyReader,
    coefficients: np.ndarray,
    official: np.ndarray,
    max_lead: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_leads = max_lead + 1
    num = np.zeros((n_leads, 3), dtype=np.float64)
    den = np.zeros((n_leads, 3), dtype=np.float64)
    t0 = time.time()
    used = 0
    for cycle_key in cycles:
        if not reader.path_for(cycle_key).is_file():
            print(f"  skip missing ENS {cycle_key}", flush=True)
            continue
        cycle_time = parse_cycle(cycle_key)
        if not truth_reader.has(cycle_time + timedelta(hours=max_lead)):
            print(f"  skip incomplete ERA5 {cycle_key}", flush=True)
            continue
        aifs = reader.get(cycle_key)
        print(f"  fit {cycle_key}", flush=True)
        used += 1
        for lead in range(n_leads):
            valid = cycle_time + timedelta(hours=lead)
            forecast = interpolate_cycle(aifs, lead)
            clim = evaluate_climatology(coefficients, valid)
            truth = truth_reader.read(valid)
            ef = forecast - truth
            diff = ef - (clim - truth)
            w = official
            num[lead] += (w * ef * diff).sum(axis=(-2, -1))
            den[lead] += (w * diff * diff).sum(axis=(-2, -1))
            if lead in (0, 48, 360) or lead % 120 == 0:
                print(f"    lead {lead:3d}  {time.time() - t0:.0f}s", flush=True)
    raw = np.divide(num, np.maximum(den, 1e-18), out=np.zeros_like(num), where=den > 1e-18)
    raw = np.clip(raw, 0.0, 1.0)
    raw[0] = 0.0
    print(f"fitted α on {used} cycles in {time.time() - t0:.0f}s", flush=True)
    return raw, smooth_alpha(raw)


def print_alpha(alpha: np.ndarray, label: str) -> None:
    print(f"α({label}) at leads", flush=True)
    for lead in (0, 24, 48, 120, 240, 360):
        row = alpha[min(lead, alpha.shape[0] - 1)]
        print(
            f"  {lead:3d}h  2t={row[0]:.3f}  u={row[1]:.3f}  v={row[2]:.3f}",
            flush=True,
        )


def empty_streams(weight_maps: dict[str, np.ndarray]) -> dict:
    return {
        h: {
            source: {name: MassStream() for name in weight_maps}
            for source in ("clim", "linear", "linear_shrunk", "cnn", "cnn_shrunk")
        }
        for h in HORIZONS
    }


def update_streams(
    streams: dict,
    lead: int,
    payloads: dict[str, np.ndarray],
    truth: np.ndarray,
    weight_maps: dict[str, np.ndarray],
) -> None:
    for horizon, by_source in streams.items():
        if lead > horizon:
            continue
        for source, prediction in payloads.items():
            for name, weights in weight_maps.items():
                by_source[source][name].update(prediction, truth, weights)


def pack_streams(streams: dict) -> dict:
    return {
        str(h): {
            source: {
                name: stream.finalize() for name, stream in by_name.items()
            }
            for source, by_name in by_source.items()
        }
        for h, by_source in streams.items()
    }


def print_cycle(cycle_key: str, packed: dict) -> None:
    for horizon in HORIZONS:
        print(f"  -- {cycle_key} {horizon}h --", flush=True)
        block = packed[str(horizon)]
        for source in ("clim", "linear", "linear_shrunk", "cnn", "cnn_shrunk"):
            scores = block[source]
            for region in ("official", "germany", "rest_europe"):
                row = scores[region]
                parts = [
                    f"{n}={row[n]['rmse']:.3f}/{row[n]['mae']:.3f}"
                    for n in SHORT_NAMES
                ]
                print(f"    {source:14s} {region:12s}  " + "  ".join(parts), flush=True)


def mean_over(results: list[dict], horizon: str, source: str, region: str) -> dict:
    out = {}
    for name in SHORT_NAMES:
        rmses = [r["horizons"][horizon][source][region][name]["rmse"] for r in results]
        maes = [r["horizons"][horizon][source][region][name]["mae"] for r in results]
        out[name] = {
            "rmse": float(np.mean(rmses)),
            "mae": float(np.mean(maes)),
            "combined": float((np.mean(rmses) + np.mean(maes)) / 2.0),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/Zeus/data/evaluation/training/aifs_downscaler_v2_ens.pt")
    parser.add_argument("--climatology", default="/Zeus/data/evaluation/training/era5_climatology_harmonics.npz")
    parser.add_argument("--ens-root", default="/Zeus/data/evaluation/aifs_ens_mean")
    parser.add_argument("--aifs-root", default="/Zeus/data/evaluation/aifs_single")
    parser.add_argument("--era5-root", default="/Zeus/data/evaluation/era5")
    parser.add_argument("--static-root", default="/Zeus/data/evaluation/training/aifs_static")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-lead", type=int, default=MAX_LEAD_HOURS)
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument(
        "--alpha-path",
        default="/Zeus/data/evaluation/training/climatology_shrinkage_alpha.npz",
    )
    parser.add_argument(
        "--output",
        default="/Zeus/data/evaluation/training/climatology_shrinkage_scores.json",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    payload = np.load(args.climatology, allow_pickle=True)
    coefficients = payload["coefficients"]
    print(
        f"climatology fit {payload['fit_start']}..{payload['fit_end']} "
        f"hours={payload['n_hours']}",
        flush=True,
    )

    latitudes = torch.linspace(-90.0, 90.0, 721)
    longitudes = torch.arange(-180.0, 180.0, 0.25)
    weight_maps = build_weight_maps(latitudes)
    reader = EnsMeanCycleReader(args.ens_root, cache_size=1)
    truth_reader = Era5HourlyReader(args.era5_root, cache_size=12)

    alpha_path = Path(args.alpha_path)
    if alpha_path.is_file():
        stored = np.load(alpha_path)
        alpha_raw = stored["alpha_raw"]
        alpha = stored["alpha_smooth"]
        print(f"loaded α from {alpha_path}", flush=True)
    else:
        print("fitting α on summer ENS cycles (no test-cycle overlap)", flush=True)
        alpha_raw, alpha = fit_alpha(
            cycles=FIT_CYCLES,
            reader=reader,
            truth_reader=truth_reader,
            coefficients=coefficients,
            official=weight_maps["official"],
            max_lead=args.max_lead,
        )
        np.savez_compressed(alpha_path, alpha_raw=alpha_raw, alpha_smooth=alpha)
        print(f"wrote {alpha_path}", flush=True)
    print_alpha(alpha_raw, "raw")
    print_alpha(alpha, "smooth")

    model = None
    statistics = None
    use_lagged = False
    single_reader = None
    land = orography = roughness = geo_feature = None
    if not args.skip_cnn:
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
        land, orography, roughness = load_static_maps(args.static_root)
        grid = get_grid(-90.0, 90.0, -180.0, 179.75)
        geo_feature = build_geographic_weights(grid, REGION_CONFIGS)
        single_reader = AifsCycleReader(args.aifs_root, cache_size=1)
        print(
            f"CNN epoch={checkpoint.get('epoch')} hidden={checkpoint['hidden_channels']} "
            f"lagged={use_lagged}",
            flush=True,
        )

    results = []
    for cycle_key in TEST_CYCLES:
        print(f"score {cycle_key}", flush=True)
        cycle_time = parse_cycle(cycle_key)
        aifs = reader.get(cycle_key)
        streams = empty_streams(weight_maps)
        t0 = time.time()
        mean = std = delta_std = residual_std = None
        if model is not None:
            mean, std, delta_std, residual_std = statistics.tensors()
        with torch.inference_mode():
            for lead in range(args.max_lead + 1):
                linear = interpolate_cycle(aifs, lead)
                clim = evaluate_climatology(coefficients, cycle_time + timedelta(hours=lead))
                truth = truth_reader.read(cycle_time + timedelta(hours=lead))
                a_lead = alpha[lead]
                linear_shrunk = blend(linear, clim, a_lead)
                payloads = {
                    "clim": clim,
                    "linear": linear,
                    "linear_shrunk": linear_shrunk,
                }
                if model is not None:
                    left, right, fraction = bracket_for_lead(lead)
                    interpolated = torch.from_numpy(linear)
                    a_left = torch.from_numpy(aifs[left].astype(np.float32))
                    a_right = torch.from_numpy(aifs[right].astype(np.float32))
                    delta = a_right - a_left
                    blocks = [(interpolated - mean) / std, delta / delta_std]
                    if use_lagged:
                        if (
                            single_reader is not None
                            and single_reader.path_for(cycle_key).is_file()
                        ):
                            prev = single_reader.get(cycle_key)
                            l2, r2, f2 = bracket_for_lead(lead)
                            lagged = (1.0 - f2) * torch.from_numpy(
                                prev[l2].astype(np.float32)
                            ) + f2 * torch.from_numpy(prev[r2].astype(np.float32))
                        else:
                            lagged = interpolated
                        blocks += [
                            (lagged - mean) / std,
                            (lagged - interpolated) / delta_std,
                        ]
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
                    cnn = (interpolated + output.correction[0] * residual_std).numpy()
                    payloads["cnn"] = cnn
                    payloads["cnn_shrunk"] = blend(cnn, clim, a_lead)
                else:
                    payloads["cnn"] = linear
                    payloads["cnn_shrunk"] = linear_shrunk
                update_streams(streams, lead, payloads, truth, weight_maps)
                if lead in (48, 360) or lead % 60 == 0:
                    print(
                        f"  {cycle_key} lead {lead:3d} {time.time() - t0:.0f}s",
                        flush=True,
                    )
        packed = pack_streams(streams)
        print_cycle(cycle_key, packed)
        results.append(
            {
                "cycle": cycle_key,
                "seconds": time.time() - t0,
                "horizons": packed,
            }
        )

    summary = {
        f"{h}h": {
            source: {
                region: mean_over(results, str(h), source, region)
                for region in weight_maps
            }
            for source in ("clim", "linear", "linear_shrunk", "cnn", "cnn_shrunk")
        }
        for h in HORIZONS
    }
    print("\n=== mean over test cycles ===", flush=True)
    for h in HORIZONS:
        print(f"{h}h", flush=True)
        for source in ("clim", "linear", "linear_shrunk", "cnn", "cnn_shrunk"):
            for region in ("official", "germany", "rest_europe"):
                row = summary[f"{h}h"][source][region]
                parts = [
                    f"{n}={row[n]['rmse']:.3f}/{row[n]['mae']:.3f}" for n in SHORT_NAMES
                ]
                print(f"  {source:14s} {region:12s}  " + "  ".join(parts), flush=True)

    Path(args.output).write_text(
        json.dumps(
            {
                "fit_cycles": list(FIT_CYCLES),
                "test_cycles": list(TEST_CYCLES),
                "alpha_path": str(alpha_path),
                "checkpoint": args.checkpoint,
                "cycles": results,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
