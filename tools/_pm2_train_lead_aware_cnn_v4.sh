#!/usr/bin/env bash
# v4: seasonal 512 tiles + 2° Earth, T/wind-heavy validator selection.
set -euo pipefail
cd /Zeus
export PYTHONPATH=/Zeus
export OMP_NUM_THREADS=8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy \
      GIT_HTTP_PROXY GIT_HTTPS_PROXY || true

PY=/root/miniconda3/envs/zeus-fourvar/bin/python
PLAN=data/evaluation/plans/cnn_residual_v4_seasonal_split.json
STATIC=data/evaluation/training/v4_static
CACHE=data/evaluation/training/lead_aware_v4_cycles
CHECKPOINT=data/evaluation/training/lead_aware_gated_residual_cnn_v4.pt
VAL_OUT=data/evaluation/results/lead_aware_residual_cnn_v4/validation.json
DIAG48=data/evaluation/results/lead_aware_residual_cnn_v4/diagnostic_20260727T000000Z_048h.json
DIAG360=data/evaluation/results/lead_aware_residual_cnn_v4/diagnostic_20260727T000000Z_360h.json

echo "PHASE 1/5: static land-sea and orography"
"$PY" tools/build_v4_static_fields.py --output-dir "$STATIC"

echo "PHASE 2/5: cache seasonal GFS/ERA5 cubes"
"$PY" tools/build_lead_aware_v4_cycle_cache.py \
  --split-plan "$PLAN" \
  --splits train,validation \
  --output-root "$CACHE"

echo "PHASE 3/5: train v4"
"$PY" zeus_ml/train/train_lead_aware_residual_cnn_v4.py \
  --split-plan "$PLAN" \
  --cache-root "$CACHE" \
  --static-root "$STATIC" \
  --output "$CHECKPOINT" \
  --epochs 15 \
  --batch-size 1 \
  --steps-per-epoch 2048 \
  --patience 3 \
  --selection-every 3 \
  --horizon 360 \
  --device cpu

echo "PHASE 4/5: in-season validation 20250922"
"$PY" zeus_ml/evaluate/evaluate_lead_aware_residual_cnn.py \
  --checkpoint "$CHECKPOINT" \
  --split-plan "$PLAN" \
  --split validation \
  --static-root "$STATIC" \
  --horizon 360 \
  --device cpu \
  --output "$VAL_OUT"

echo "PHASE 5/5: frozen 2026 diagnostic 48h and 360h"
"$PY" zeus_ml/evaluate/evaluate_lead_aware_residual_cnn.py \
  --checkpoint "$CHECKPOINT" \
  --split-plan "$PLAN" \
  --split diagnostic \
  --cycle 20260727T000000Z \
  --horizon 48 \
  --static-root "$STATIC" \
  --device cpu \
  --output "$DIAG48"
"$PY" zeus_ml/evaluate/evaluate_lead_aware_residual_cnn.py \
  --checkpoint "$CHECKPOINT" \
  --split-plan "$PLAN" \
  --split diagnostic \
  --cycle 20260727T000000Z \
  --horizon 360 \
  --static-root "$STATIC" \
  --device cpu \
  --output "$DIAG360"

echo "LEAD_AWARE_CNN_V4_PIPELINE_COMPLETE"
