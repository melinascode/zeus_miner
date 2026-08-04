#!/usr/bin/env bash
# Rebuild all locked test GFS bundles with production source offsets (6/12/18/24).
set -euo pipefail
cd /Zeus
export PYTHONPATH=/Zeus
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy \
      GIT_HTTP_PROXY GIT_HTTPS_PROXY || true

# Historical AWS archives are complete for these targets; offset 6 is always the
# newest ready production choice. Still verified against the manifest after each build.
exec /root/miniconda3/envs/zeus-fourvar/bin/python \
  tools/rebuild_hist_gfs_production_faithful.py \
  --assume-offset 6 \
  --hotkey 5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR
