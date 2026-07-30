#!/usr/bin/env bash
# Retry only the two failed benchmark_v1 GFS test cycles.
set -euo pipefail
cd /Zeus
export PYTHONPATH=/Zeus
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy \
      GIT_HTTP_PROXY GIT_HTTPS_PROXY || true

PY=/root/miniconda3/envs/zeus-fourvar/bin/python
HOTKEY=5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR
STATUS=data/evaluation/plans/benchmark_v1_download_status.json

# Reset failed -> pending for the two targets only.
"$PY" - <<'PY'
import json
from pathlib import Path
path = Path("data/evaluation/plans/benchmark_v1_download_status.json")
payload = json.loads(path.read_text())
for cycle in ("20250922T060000Z", "20260623T180000Z"):
    entry = payload["gfs"].setdefault(cycle, {})
    entry["status"] = "pending"
    entry["failure_reason"] = None
    entry.pop("elapsed_seconds", None)
    entry.pop("started_at_utc", None)
    entry.pop("finished_at_utc", None)
    print(f"reset {cycle} -> pending")
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

for CYCLE in 20250922T060000Z 20260623T180000Z; do
  echo "RETRY START $CYCLE"
  "$PY" tools/download_benchmark_v1_test_data.py \
    --skip-era5 \
    --gfs-only-cycle "$CYCLE" \
    --hotkey "$HOTKEY"
  echo "RETRY DONE $CYCLE"
done

echo "RETRY BATCH FINISHED"
"$PY" - <<'PY'
import json
from pathlib import Path
d=json.loads(Path("data/evaluation/plans/benchmark_v1_download_status.json").read_text())
for c in ("20250922T060000Z","20260623T180000Z"):
    print(c, d["gfs"][c])
PY
