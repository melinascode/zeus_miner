#!/usr/bin/env bash
# Diagnostic: native GFS + ERA5 + v3 CNN scores for 20260727T000000Z
# at 48h (2-day) and 360h (15-day). Writes only under data/evaluation/.
set -euo pipefail
cd /Zeus
export PYTHONPATH=/Zeus
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
      SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy \
      GIT_HTTP_PROXY GIT_HTTPS_PROXY || true

PY=/root/miniconda3/envs/zeus-fourvar/bin/python
CYCLE=20260727T000000Z
CHECKPOINT=data/evaluation/training/lead_aware_gated_residual_cnn_v3.pt
PLAN=data/evaluation/plans/cnn_residual_v3_user_split.json
BUNDLE=data/evaluation/forecast_store_hist/bundles/${CYCLE}
OUT_DIR=data/evaluation/results/lead_aware_residual_cnn_v3
HOTKEY=5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR

echo "PHASE 1/3: ERA5 and historical GFS in parallel for ${CYCLE}"

"$PY" tools/fetch_era5_google_evaluation.py \
  --start-date 2026-07-27 \
  --end-date 2026-08-11 \
  --output-dir data/evaluation/era5 &
ERA5_PID=$!

"$PY" tools/build_historical_gfs_bundle.py \
  --target-cycle "${CYCLE}" \
  --hotkey "${HOTKEY}" \
  --store-dir data/evaluation/forecast_store_hist \
  --cache-dir data/evaluation/gfs_cache \
  --work-dir "data/evaluation/gfs_work/${CYCLE}" &
GFS_PID=$!

ERA5_STATUS=0
GFS_STATUS=0
wait "${ERA5_PID}" || ERA5_STATUS=$?
wait "${GFS_PID}" || GFS_STATUS=$?
echo "ERA5_STATUS=${ERA5_STATUS} GFS_STATUS=${GFS_STATUS}"
if [[ "${ERA5_STATUS}" -ne 0 || "${GFS_STATUS}" -ne 0 ]]; then
  echo "CNN_V3_20260727_PIPELINE_FAILED"
  exit 1
fi
if [[ ! -d "${BUNDLE}" ]]; then
  echo "Missing GFS bundle ${BUNDLE}"
  echo "CNN_V3_20260727_PIPELINE_FAILED"
  exit 1
fi

echo "PHASE 2/3: score 48-hour (2-day) challenge"
"$PY" zeus_ml/evaluate/evaluate_lead_aware_residual_cnn.py \
  --checkpoint "${CHECKPOINT}" \
  --split-plan "${PLAN}" \
  --split diagnostic \
  --cycle "${CYCLE}" \
  --horizon 48 \
  --device cpu \
  --output "${OUT_DIR}/diagnostic_${CYCLE}_048h.json"

echo "PHASE 3/3: score 360-hour (15-day) challenge"
"$PY" zeus_ml/evaluate/evaluate_lead_aware_residual_cnn.py \
  --checkpoint "${CHECKPOINT}" \
  --split-plan "${PLAN}" \
  --split diagnostic \
  --cycle "${CYCLE}" \
  --horizon 360 \
  --device cpu \
  --output "${OUT_DIR}/diagnostic_${CYCLE}_360h.json"

echo "CNN_V3_20260727_PIPELINE_COMPLETE"
