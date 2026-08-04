#!/usr/bin/env bash
# Wait for GFS smoke cycle to finish, then run remaining 29 test cycles.
set -euo pipefail
cd /Zeus
export PYTHONPATH=/Zeus
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy \
      GIT_HTTP_PROXY GIT_HTTPS_PROXY || true

STATUS=data/evaluation/plans/benchmark_v1_download_status.json
echo "Waiting for smoke cycle 20250422T180000Z ..."
while true; do
  if [[ -f "$STATUS" ]]; then
    state=$(/root/miniconda3/envs/zeus-fourvar/bin/python - <<'PY'
import json
from pathlib import Path
d=json.loads(Path("data/evaluation/plans/benchmark_v1_download_status.json").read_text())
print(d.get("gfs",{}).get("20250422T180000Z",{}).get("status","missing"))
PY
)
    echo "smoke_status=$state"
    if [[ "$state" == "complete" || "$state" == "failed" ]]; then
      break
    fi
  fi
  sleep 60
done

echo "Starting full 30-cycle GFS download (skips completed cycles) ..."
exec /root/miniconda3/envs/zeus-fourvar/bin/python tools/download_benchmark_v1_test_data.py \
  --skip-era5 \
  --hotkey 5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR
