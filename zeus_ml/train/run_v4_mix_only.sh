#!/usr/bin/env bash
# V4 mix-head-only. Eval/training only — does not touch the live miner.
#
# Follow-up to v4_wind_sharp: that run proved gradient/speed/wind-heavy
# weights make long-lead 100u/100v worse. This run freezes the V3 trunk and
# trains only the zero-init temporal mix heads, with those wind terms off,
# so the mid-hour interpolator can move without wrecking V3's t2m solution.
#
# Baseline to beat: V3 val t2m +4.68%, u100 -0.80%, v100 -0.10%. Keep the
# checkpoint only if weighted gain >= V3 and wind is not worse.
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

NAME="${1:-aifs_downscaler_v4_mix_only}"

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
  --mix-head-only \
  --geo-mode official \
  --mae-weight 0.65 \
  --midhour-boost 1.5 \
  --gradient-weight 0 \
  --speed-weight 0 \
  --wind-no-regret-weight 0 \
  --long-lead-boost 0 \
  --no-regret-weight 0.5 \
  --select-on "${SELECT_ON:-weighted}" \
  --grib-cache-size 32 \
  --epochs 8 \
  --cycles-per-epoch 24 \
  --leads-per-cycle 24 \
  --tile-size 192 \
  --tiles-per-item 6 \
  --learning-rate 2e-4 \
  --europe-fraction 0.50 \
  --device cuda \
  --threads 8
