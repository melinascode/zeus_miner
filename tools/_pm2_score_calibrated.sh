#!/usr/bin/env bash
set -euo pipefail
cd /Zeus
export PYTHONPATH=/Zeus
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy || true
# Seed the calibrated registry from the locked test registry if missing.
REG=data/evaluation/plans/benchmark_v1_registry_calibrated.json
if [[ ! -f "$REG" ]]; then
  python3 - <<'PY'
import json, shutil
from pathlib import Path
src = Path("data/evaluation/plans/benchmark_v1_registry.json")
dst = Path("data/evaluation/plans/benchmark_v1_registry_calibrated.json")
payload = json.loads(src.read_text())
payload["amendment_note"] = (
    "Calibrated-GFS vs raw/persistence scoring; seeded from locked test registry."
)
for cycle in payload.get("cycles", {}).values():
    cycle["status"] = "pending"
    cycle["failure_reason"] = None
    cycle["updated_at_utc"] = None
dst.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"seeded {dst}")
PY
fi
exec /root/miniconda3/envs/zeus-fourvar/bin/python tools/run_evaluation_backtest.py \
  --plan data/evaluation/plans/benchmark_v1_backtest_plan.json \
  --selection data/evaluation/plans/benchmark_v1_selection.json \
  --registry data/evaluation/plans/benchmark_v1_registry_calibrated.json \
  --coefficients data/evaluation/plans/benchmark_v1_calibrated_gfs_coefficients.json \
  --store-dir data/evaluation/forecast_store_hist \
  --output-dir data/evaluation/results/calibrated_vs_raw \
  --expected-hotkey 5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR \
  --minimum-cycles 30 \
  --require-full-matrix \
  --require-independent \
  --lead-diagnostics \
  --continue-on-error \
  --wandb \
  --wandb-mode online
