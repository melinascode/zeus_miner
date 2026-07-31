#!/usr/bin/env bash
set -euo pipefail
cd /Zeus
export PYTHONPATH=/Zeus
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy || true
exec /root/miniconda3/envs/zeus-fourvar/bin/python tools/run_evaluation_backtest.py \
  --plan data/evaluation/plans/benchmark_v1_backtest_plan.json \
  --selection data/evaluation/plans/benchmark_v1_selection.json \
  --registry data/evaluation/plans/benchmark_v1_registry.json \
  --store-dir data/evaluation/forecast_store_hist \
  --output-dir data/evaluation/results/raw_vs_persistence \
  --expected-hotkey 5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR \
  --minimum-cycles 30 \
  --require-full-matrix \
  --require-independent \
  --continue-on-error
