#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

ENV_FILE="$SCRIPT_DIR/miner.env"
PROCESS_NAME="zeus_bundle_builder"
PYTHON_BIN="/root/miniconda3/envs/zeus-fourvar/bin/python"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: miner.env not found at $ENV_FILE"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

for variable in WALLET_NAME WALLET_HOTKEY; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Error: $variable is empty or missing in miner.env"
    exit 1
  fi
done

echo "Starting bundle builder:"
echo "  wallet: $WALLET_NAME / $WALLET_HOTKEY"
echo "  python: $PYTHON_BIN"

pm2 delete "$PROCESS_NAME" >/dev/null 2>&1 || true

pm2 start "$SCRIPT_DIR/zeus_ml/serve/live_bundle_builder.py" \
  --name "$PROCESS_NAME" \
  --interpreter "$PYTHON_BIN"

pm2 save
pm2 status
