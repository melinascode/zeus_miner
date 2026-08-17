#!/usr/bin/env bash
set -euo pipefail

cd /Zeus
export PYTHONPATH=/Zeus
export OMP_NUM_THREADS=8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy || true

PY=/root/miniconda3/envs/zeus-fourvar/bin/python
PLAN=data/evaluation/plans/cnn_residual_v3_user_split.json
PATCH_ROOT=data/evaluation/training/lead_aware_patches_v3
CHECKPOINT=data/evaluation/training/lead_aware_gated_residual_cnn_v3.pt
VAL_RESULT=data/evaluation/results/lead_aware_residual_cnn_v3/validation.json
TEST_RESULT=data/evaluation/results/lead_aware_residual_cnn_v3/test.json

echo "PHASE 1/5: build training patches with region and long-lead sampling"
"$PY" tools/build_lead_aware_patch_dataset.py \
  --split-plan "$PLAN" \
  --split train \
  --output-root "$PATCH_ROOT" \
  --patch-size 128 \
  --patches-per-lead 6

echo "PHASE 2/5: build validation patches"
"$PY" tools/build_lead_aware_patch_dataset.py \
  --split-plan "$PLAN" \
  --split validation \
  --output-root "$PATCH_ROOT" \
  --patch-size 128 \
  --patches-per-lead 6

echo "PHASE 3/5: train v3 CNN with validator-faithful selection"
"$PY" zeus_ml/train/train_lead_aware_residual_cnn.py \
  --split-plan "$PLAN" \
  --patch-root "$PATCH_ROOT" \
  --output "$CHECKPOINT" \
  --epochs 30 \
  --batch-size 4 \
  --hidden-channels 32 \
  --patience 3 \
  --selection-every 3 \
  --selection-max-cycles 4 \
  --horizon 360 \
  --device cpu

echo "PHASE 4/5: full-tensor validation selection score"
"$PY" zeus_ml/evaluate/evaluate_lead_aware_residual_cnn.py \
  --checkpoint "$CHECKPOINT" \
  --split-plan "$PLAN" \
  --split validation \
  --output "$VAL_RESULT" \
  --horizon 360 \
  --device cpu

echo "PHASE 5/5: held-out test (not used for selection)"
"$PY" zeus_ml/evaluate/evaluate_lead_aware_residual_cnn.py \
  --checkpoint "$CHECKPOINT" \
  --split-plan "$PLAN" \
  --split test \
  --output "$TEST_RESULT" \
  --horizon 360 \
  --device cpu

echo "LEAD_AWARE_CNN_V3_PIPELINE_COMPLETE"
