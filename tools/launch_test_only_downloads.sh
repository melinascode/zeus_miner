#!/usr/bin/env bash
# Launch test-only ERA5 + GFS downloads for locked benchmark_v1.
# Run from a normal host shell (not the Cursor agent sandbox).
set -euo pipefail
cd /Zeus

mkdir -p data/evaluation/logs data/evaluation/era5 \
  data/evaluation/forecast_store_hist data/evaluation/gfs_cache data/evaluation/gfs_work

PY=/root/miniconda3/envs/zeus-fourvar/bin/python
export PYTHONPATH=/Zeus
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy \
      GIT_HTTP_PROXY GIT_HTTPS_PROXY || true

echo "=== network preflight ==="
getent hosts cds.climate.copernicus.eu noaa-gfs-bdp-pds.s3.amazonaws.com || true
curl -sI --max-time 15 https://cds.climate.copernicus.eu/api | head -3 || true
curl -sI --max-time 15 https://noaa-gfs-bdp-pds.s3.amazonaws.com/ | head -3 || true

MODE="${1:-both}"  # era5 | gfs | both

if [[ "$MODE" == "era5" || "$MODE" == "both" ]]; then
  echo "=== starting ERA5 test-window download (2025-04-22 → 2026-07-24) ==="
  nohup "$PY" tools/fetch_era5_evaluation.py \
    --start-date 2025-04-22 \
    --end-date 2026-07-24 \
    --env-file validator.env \
    > data/evaluation/logs/era5_test_download.log 2>&1 &
  echo "ERA5_PID=$!"
  echo "log: data/evaluation/logs/era5_test_download.log"
fi

if [[ "$MODE" == "gfs" || "$MODE" == "both" ]]; then
  echo "=== starting 30-cycle historical GFS download/build ==="
  nohup "$PY" tools/download_benchmark_v1_test_data.py \
    --skip-era5 \
    --hotkey 5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR \
    > data/evaluation/logs/gfs_test_download.log 2>&1 &
  echo "GFS_PID=$!"
  echo "log: data/evaluation/logs/gfs_test_download.log"
  echo "status: data/evaluation/plans/benchmark_v1_download_status.json"
fi

echo "=== launched ==="
echo "Monitor with:"
echo "  tail -f data/evaluation/logs/era5_test_download.log"
echo "  tail -f data/evaluation/logs/gfs_test_download.log"
echo "  watch -n 30 'python - <<\"PY\"
import json
from pathlib import Path
p=Path(\"data/evaluation/plans/benchmark_v1_download_status.json\")
print(\"era5 files\", sum(1 for _ in Path(\"data/evaluation/era5\").rglob(\"*.nc\")))
if p.exists():
  d=json.loads(p.read_text())
  g=d.get(\"gfs\",{})
  print(\"gfs\", {s:sum(1 for v in g.values() if v.get(\"status\")==s) for s in (\"pending\",\"running\",\"complete\",\"failed\")})
PY'"
