#!/usr/bin/env bash
set -euo pipefail
cd /Zeus
export PYTHONPATH=/Zeus
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy \
      GIT_HTTP_PROXY GIT_HTTPS_PROXY || true
exec /root/miniconda3/envs/zeus-fourvar/bin/python tools/download_benchmark_v1_test_data.py \
  --skip-era5 \
  --gfs-only-cycle 20250422T180000Z \
  --hotkey 5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR
