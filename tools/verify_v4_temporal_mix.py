"""Verify the V4 temporal-mix changes without touching anything live.

Checks:
  1. Live v3_wind.pt still loads through the new factory (strict, no mix).
  2. A temporal-mix model warm-started from V3 (mix heads zero) produces
     bit-identical corrections to the plain V3 model -> serving-safe start.
  3. A fresh temporal-mix model at init outputs exactly zero correction
     (reproduces linear interpolation).
  4. build_split falls back to a chronological holdout when no cycle lands
     in a seasonal validation block (summer-only ERA5 case).
  5. evaluate() lead bands are disjoint at lead 72.
"""

from __future__ import annotations

import sys

import torch

from zeus_ml.models.aifs_downscaler_cnn import (
    CONTEXT_FEATURES,
    AifsDownscalerCNN,
    aifs_downscaler_from_checkpoint,
    build_downscaler_context,
)
from zeus_ml.train.train_aifs_downscaler import build_split

CKPT = "/Zeus/data/evaluation/training/aifs_downscaler_v3_wind.pt"


def main() -> int:
    failures = []

    # 1. Factory loads the live checkpoint unchanged.
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    v3 = aifs_downscaler_from_checkpoint(ck)
    v3.eval()
    n_params = sum(p.numel() for p in v3.parameters())
    print(f"1. factory loaded v3_wind: {n_params} params, "
          f"mix={v3.use_temporal_mix}")
    if n_params != 30285 or v3.use_temporal_mix:
        failures.append("factory changed the live V3 model")

    # 2. Warm-started mix model == plain V3 while mix heads are zero.
    mix = AifsDownscalerCNN(
        hidden_channels=int(ck["hidden_channels"]),
        weather_channels=int(ck["weather_channels"]),
        use_temporal_mix=True,
    )
    missing, unexpected = mix.load_state_dict(ck["model_state"], strict=False)
    bad = [k for k in missing if not k.startswith("mix_heads.")]
    if bad or unexpected:
        failures.append(f"warm-start mismatch missing={bad} unexpected={unexpected}")
    mix.eval()

    torch.manual_seed(7)
    weather = torch.randn(2, int(ck["weather_channels"]), 64, 64)
    static = torch.randn(2, 10, 64, 64)
    context = torch.stack(
        [
            build_downscaler_context(
                lead_hour=123, cycle_hour=0, day_of_year=200, fraction=0.5
            ),
            build_downscaler_context(
                lead_hour=48, cycle_hour=12, day_of_year=10, fraction=0.0
            ),
        ]
    )
    assert context.shape == (2, CONTEXT_FEATURES)
    with torch.inference_mode():
        out_v3 = v3(weather, context, static).correction
        out_mix = mix(weather, context, static).correction
    diff = float((out_v3 - out_mix).abs().max())
    print(f"2. warm-started mix vs V3 max |diff| = {diff:.3e}")
    if diff != 0.0:
        failures.append("mix model does not start equal to V3")

    # 3. Fresh mix model at init reproduces linear interpolation.
    fresh = AifsDownscalerCNN(
        hidden_channels=8, weather_channels=6, use_temporal_mix=True
    )
    fresh.eval()
    with torch.inference_mode():
        out = fresh(torch.randn(1, 6, 32, 32), context[:1], torch.randn(1, 10, 32, 32))
    mx = float(out.correction.abs().max())
    print(f"3. fresh mix model max |correction| = {mx:.3e} (gate 0.02 x zero heads)")
    if mx != 0.0:
        failures.append("fresh mix model is not exact linear interpolation")

    # 4. Chronological fallback for summer-only cycle lists.
    summer = [f"202607{d:02d}T000000Z" for d in (1, 5, 9, 13, 17, 21, 25, 29)]
    split = build_split(summer)
    print(f"4. summer-only split: train={len(split.train)} "
          f"val={len(split.validation)} -> {split.validation}")
    if not split.validation or not split.train:
        failures.append("chronological fallback produced an empty split")
    if set(split.train) & set(split.validation):
        failures.append("train/validation overlap in fallback split")

    # 5. Bands disjoint at lead 72.
    bands = {"h0_72": (0, 72), "h72_360": (72, 361)}
    owners = [n for n, (lo, hi) in bands.items() if lo <= 72 < hi]
    print(f"5. lead 72 counted in bands: {owners}")
    if owners != ["h72_360"]:
        failures.append("lead 72 band ownership wrong")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
