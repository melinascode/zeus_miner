#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

ENV_FILE="$SCRIPT_DIR/miner.env"
MINER_PROCESS_NAME="zeus_miner"
PYTHON_BIN="/root/miniconda3/envs/zeus-fourvar/bin/python"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: miner.env not found at $ENV_FILE"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python not found at $PYTHON_BIN"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

required_variables=(
  NETUID
  SUBTENSOR_NETWORK
  SUBTENSOR_CHAIN_ENDPOINT
  WALLET_NAME
  WALLET_HOTKEY
  AXON_PORT
  BLACKLIST_FORCE_VALIDATOR_PERMIT
)

for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Error: $variable is empty or missing in miner.env"
    exit 1
  fi
done

echo "Starting miner:"
echo "  netuid:  $NETUID"
echo "  network: $SUBTENSOR_NETWORK"
echo "  wallet:  $WALLET_NAME"
echo "  hotkey:  $WALLET_HOTKEY"
echo "  port:    $AXON_PORT"
echo "  python:  $PYTHON_BIN"

pm2 delete "$MINER_PROCESS_NAME" >/dev/null 2>&1 || true

pm2 start "$SCRIPT_DIR/neurons/miner.py" \
  --name "$MINER_PROCESS_NAME" \
  --interpreter "$PYTHON_BIN" \
  -- \
  --netuid "$NETUID" \
  --subtensor.network "$SUBTENSOR_NETWORK" \
  --subtensor.chain_endpoint "$SUBTENSOR_CHAIN_ENDPOINT" \
  --wallet.name "$WALLET_NAME" \
  --wallet.hotkey "$WALLET_HOTKEY" \
  --axon.port "$AXON_PORT" \
  --blacklist.force_validator_permit "$BLACKLIST_FORCE_VALIDATOR_PERMIT" \
  --logging.info

pm2 save
pm2 status