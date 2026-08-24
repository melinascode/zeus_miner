"""Score one cycle: linear interpolation, optional downscaler, 48h/360h iwRMSE/iwMAE."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from zeus_ml.datasets.aifs_downscale_dataset import (
    AifsCycleReader,
    EnsMeanCycleReader,
    Era5HourlyReader,
)
from zeus_ml.evaluate.evaluate_aifs_downscaler import (
    SHORT_NAMES,
    Stream,
    cycle_maps,
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


def fmt(final: list[dict]) -> str:
    return "  ".join(
        f"{n}={final[i]['rmse']:.3f}/{final[i]['mae']:.3f}"
        for i, n in enumerate(SHORT_NAMES)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--ens-root", default=None)
    parser.add_argument("--aifs-root", default="/Zeus/data/evaluation/aifs_single")
    parser.add_argument("--era5-root", default="/Zeus/data/evaluation/era5")
    parser.add_argument("--static-root", default="/Zeus/data/evaluation/training/aifs_static")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    cycle = args.cycle
    cycle_time = datetime.strptime(cycle, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    latitudes = torch.linspace(-90.0, 90.0, 721)
    longitudes = torch.arange(-180.0, 180.0, 0.25)
    _, weights = cycle_maps(cycle_time, latitudes)
    truth = Era5HourlyReader(args.era5_root)
    if args.ens_root:
        reader = EnsMeanCycleReader(args.ens_root, cache_size=1)
        single_reader = AifsCycleReader(args.aifs_root, cache_size=1)
        source = "ENS-mean"
    else:
        reader = AifsCycleReader(args.aifs_root, cache_size=1)
        single_reader = None
        source = "AIFS Single"
    label = args.label or source
    fields = reader.get(cycle)

    model = None
    residual_std = None
    mean = std = delta_std = None
    land = orography = roughness = None
    use_lagged = False
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        statistics = DownscalerStatistics.from_dict(checkpoint["statistics"])
        mean, std, delta_std, residual_std = statistics.tensors()
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

    linear = {48: Stream(), 360: Stream()}
    cnn = {48: Stream(), 360: Stream()} if model is not None else None
    with torch.inference_mode():
        for lead in range(MAX_LEAD_HOURS + 1):
            left, right, fraction = bracket_for_lead(lead)
            interpolated = (1.0 - fraction) * fields[left].astype(
                np.float32
            ) + fraction * fields[right].astype(np.float32)
            pred = torch.from_numpy(interpolated)
            truth_hour = torch.from_numpy(
                truth.read(cycle_time + timedelta(hours=lead))
            )
            for horizon, stream in linear.items():
                if lead <= horizon:
                    stream.update(pred, truth_hour, weights)
            if model is not None:
                delta = torch.from_numpy(
                    fields[right].astype(np.float32) - fields[left].astype(np.float32)
                )
                blocks = [(pred - mean) / std, delta / delta_std]
                if use_lagged:
                    pair_reader = single_reader if single_reader is not None else reader
                    if single_reader is not None:
                        prev_key = cycle
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
                        lagged = pred
                    blocks += [(lagged - mean) / std, (lagged - pred) / delta_std]
                weather = torch.cat(blocks, dim=0).unsqueeze(0)
                zenith, zenith_anomaly = zenith_triplet(
                    latitudes, longitudes, cycle_time, lead
                )
                geographic, _ = cycle_maps(cycle_time, latitudes)
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
                corrected = pred + output.correction[0] * residual_std
                for horizon, stream in cnn.items():
                    if lead <= horizon:
                        stream.update(corrected, truth_hour, weights)
            if lead in (48, 360) or lead % 120 == 0:
                print(f"  {label} lead {lead}", flush=True)

    print(f"{cycle} {label}")
    for horizon in (48, 360):
        print(f"  {horizon}h linear {fmt(linear[horizon].finalize())}")
        if cnn is not None:
            print(f"  {horizon}h cnn    {fmt(cnn[horizon].finalize())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
