#!/bin/bash
# Run on the Runpod 4090 only. Does not touch the live miner.
set -euo pipefail
cd /workspace/zeus
export PYTHONPATH=/workspace/zeus
PY=/workspace/venv/bin/python
exec "$PY" -u -m zeus_ml.train.train_aifs_downscaler \
  --ens-root /workspace/data/evaluation/aifs_ens_mean \
  --aifs-root /workspace/data/evaluation/aifs_single \
  --era5-root /workspace/data/evaluation/era5 \
  --static-root /workspace/data/evaluation/training/aifs_static \
  --output-root /workspace/data/evaluation/training \
  --name aifs_downscaler_v3_wind \
  --init-from /workspace/data/evaluation/training/aifs_downscaler_v2_ens.pt \
  --statistics-from /workspace/data/evaluation/training/aifs_downscaler_v2_ens.statistics.json \
  --hidden-channels 48 \
  --use-lagged \
  --geo-mode official \
  --mae-weight 0.65 \
  --midhour-boost 1.0 \
  --europe-fraction 0.50 \
  --epochs 8 \
  --cycles-per-epoch 24 \
  --leads-per-cycle 24 \
  --learning-rate 1e-3 \
  --device cuda \
  --threads 8
