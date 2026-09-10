#!/usr/bin/env bash
# V4 wind-sharp downscaler. Eval/training only — does not touch the live
# miner, bundle builder, or forecast_store_v2.
#
# Same inputs as live V3 (ENS-mean + lagged AIFS-single, hidden=48).
# Changes vs V3: stronger mid-hour oversampling, MAE-heavier combined loss,
# spatial gradient sharpness, a fraction-aware temporal head (zero-init,
# 4f(1-f)-gated term on the 6h tendency so mid-hours are predicted from
# both anchors instead of pure linear interpolation), plus wind terms:
# selection/loss weights tilted to 100u/100v, oversampling of leads >72h
# (where the 361h error mass sits) and a 100m wind-speed MAE term against
# ensemble-mean |V| damping. Warm-start from v3_wind.pt.
#
# Baseline to beat: V3 validation gain u100 -0.80%, v100 -0.10% (the live
# CNN does not help wind). Any V4 with u/v gains > 0 is already progress.
#
# RTX 4090 24GB is enough. Do not move the US-IL-1 volume for an A100.
set -euo pipefail

if [[ -d /workspace/zeus && -d /workspace/data ]]; then
  ROOT=/workspace/zeus
  DATA=/workspace/data
  PYTHON="${PYTHON:-/workspace/venv/bin/python}"
else
  ROOT="${ZEUS_ROOT:-/Zeus}"
  DATA="${DATA_ROOT:-$ROOT/data}"
  PYTHON="${PYTHON:-/root/miniconda3/envs/zeus-fourvar/bin/python}"
fi

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

NAME="${1:-aifs_downscaler_v4_wind_sharp}"

exec "$PYTHON" -u -m zeus_ml.train.train_aifs_downscaler \
  --aifs-root "${AIFS_ROOT:-$DATA/evaluation/aifs_single}" \
  --ens-root "${ENS_ROOT:-$DATA/evaluation/aifs_ens_mean}" \
  --era5-root "${ERA5_ROOT:-$DATA/evaluation/era5}" \
  --static-root "${STATIC_ROOT:-$DATA/evaluation/training/aifs_static}" \
  --output-root "${OUTPUT_ROOT:-$DATA/evaluation/training}" \
  --name "$NAME" \
  --init-from "${INIT_FROM:-$DATA/evaluation/training/aifs_downscaler_v3_wind.pt}" \
  --statistics-from "${STATS_FROM:-$DATA/evaluation/training/aifs_downscaler_v2_ens.statistics.json}" \
  --hidden-channels 48 \
  --use-lagged \
  --use-temporal-mix \
  --geo-mode official \
  --mae-weight 0.65 \
  --midhour-boost 1.5 \
  --gradient-weight 0.25 \
  --no-regret-weight 0.5 \
  --variable-weights "${VAR_WEIGHTS:-0.20,0.40,0.40}" \
  --long-lead-boost "${LONG_LEAD_BOOST:-1.0}" \
  --long-lead-start 72 \
  --speed-weight "${SPEED_WEIGHT:-0.15}" \
  --wind-no-regret-weight "${WIND_NO_REGRET:-0.5}" \
  --select-on "${SELECT_ON:-wind_long}" \
  --epochs 10 \
  --cycles-per-epoch 24 \
  --leads-per-cycle 24 \
  --tile-size 192 \
  --tiles-per-item 6 \
  --learning-rate 8e-4 \
  --europe-fraction 0.50 \
  --device cuda \
  --threads 8
