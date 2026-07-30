#!/usr/bin/env python3
"""Log an existing evaluator JSON result from the zeus-eval environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.wandb_logging import log_evaluation_to_wandb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    args = parser.parse_args()

    result_path = Path(args.result).resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_reference = log_evaluation_to_wandb(
        result,
        result_path,
        mode=args.mode,
    )
    print(json.dumps({"wandb_run": run_reference}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
