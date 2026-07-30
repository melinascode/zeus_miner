#!/usr/bin/env bash
set -euo pipefail
cd /Zeus
export PYTHONPATH=/Zeus
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy \
      GIT_HTTP_PROXY GIT_HTTPS_PROXY || true
exec /root/miniconda3/envs/zeus-fourvar/bin/python tools/fetch_era5_google_evaluation.py \
  --start-date 2025-04-22 \
  --end-date 2026-07-24
