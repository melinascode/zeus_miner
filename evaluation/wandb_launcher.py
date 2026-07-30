from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WANDB_PYTHON = Path("/root/miniconda3/envs/zeus-eval/bin/python")


def launch_wandb_logger(
    result_path: str | Path,
    mode: str,
) -> dict[str, str]:
    """Launch W&B logging in the dedicated minimal environment."""

    if not WANDB_PYTHON.is_file():
        raise FileNotFoundError(
            f"Required W&B interpreter does not exist: {WANDB_PYTHON}"
        )
    completed = subprocess.run(
        [
            str(WANDB_PYTHON),
            str(PROJECT_ROOT / "tools" / "log_evaluation_wandb.py"),
            "--result",
            str(Path(result_path).resolve()),
            "--mode",
            mode,
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in reversed(completed.stdout.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "wandb_run" in parsed:
            return parsed
    raise RuntimeError(
        "Dedicated W&B logger returned no result JSON. "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
