#!/usr/bin/env bash
set -euo pipefail

cd /Zeus
export PYTHONPATH=/Zeus
export OMP_NUM_THREADS=8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy || true

PY=/root/miniconda3/envs/zeus-fourvar/bin/python
PLAN=data/evaluation/plans/cnn_residual_v2_development_split.json
PATCH_ROOT=data/evaluation/training/lead_aware_patches
CHECKPOINT=data/evaluation/training/lead_aware_gated_residual_cnn_v2.pt
RESULT=data/evaluation/results/lead_aware_residual_cnn_v2/evaluation.json

echo "PHASE 1/4: build training patches"
"$PY" tools/build_lead_aware_patch_dataset.py \
  --split-plan "$PLAN" \
  --split train \
  --output-root "$PATCH_ROOT" \
  --patch-size 128 \
  --patches-per-lead 4

echo "PHASE 2/4: build validation patches"
"$PY" tools/build_lead_aware_patch_dataset.py \
  --split-plan "$PLAN" \
  --split validation \
  --output-root "$PATCH_ROOT" \
  --patch-size 128 \
  --patches-per-lead 4

echo "PHASE 3/4: train lead-aware gated residual CNN"
"$PY" zeus_ml/train/train_lead_aware_residual_cnn.py \
  --split-plan "$PLAN" \
  --patch-root "$PATCH_ROOT" \
  --output "$CHECKPOINT" \
  --epochs 20 \
  --batch-size 4 \
  --hidden-channels 32 \
  --patience 4 \
  --device cpu

echo "PHASE 4/4: validator-faithful development test"
"$PY" zeus_ml/evaluate/evaluate_lead_aware_residual_cnn.py \
  --checkpoint "$CHECKPOINT" \
  --split-plan "$PLAN" \
  --split test \
  --output "$RESULT" \
  --horizon 360 \
  --device cpu

echo "LEAD_AWARE_CNN_PIPELINE_COMPLETE"
