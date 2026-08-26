"""Train the Germany ResUNet specialist.

Thin wrapper around train_europe_resunet with Germany domain defaults.
Uses the existing Europe crop dataset; the 80x96 Germany+context window is
sliced in memory (42-61.75N, 2W-21.75E). Loss is the official capacity
scalars restricted to the Germany scoring box (47-56N, 6-15E).

  python -m zeus_ml.train.train_germany_resunet --source single --name germany_resunet_pre
  python -m zeus_ml.train.train_germany_resunet --source ens --init-from ..._pre.pt --name germany_resunet_ens
"""

from __future__ import annotations

import sys

from zeus_ml.train.train_europe_resunet import main as europe_main


def main() -> int:
    injected = ["--domain", "germany", "--loss-region", "germany"]
    if "--name" not in sys.argv:
        injected += ["--name", "germany_resunet_v1"]
    if "--batch-size" not in sys.argv:
        # 80x96 window is ~9x smaller than the Europe crop; larger batches
        # keep the GPU fed and stabilize the gate statistics.
        injected += ["--batch-size", "64"]
    sys.argv = [sys.argv[0], *injected, *sys.argv[1:]]
    return europe_main()


if __name__ == "__main__":
    raise SystemExit(main())
