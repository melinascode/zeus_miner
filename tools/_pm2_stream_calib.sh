#!/usr/bin/env bash
set -euo pipefail
cd /Zeus
export PYTHONPATH=/Zeus
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy || true
exec /root/miniconda3/envs/zeus-fourvar/bin/python tools/run_streaming_calibration.py \
  --resume \
  --assume-offset 6 \
  --hotkey 5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR
