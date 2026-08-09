from pathlib import Path
from datetime import datetime, timezone

import torch

from zeus_ml.features.gfs_loader import load_gfs_artifact


# -------------------------
# Configuration
# -------------------------

BUNDLE = Path(
    "/Zeus/data/evaluation/forecast_store_hist/bundles/20250201T000000Z"
)

OUT = Path(
    "/Zeus/data/evaluation/residual_ml/holdout"
)

VARIABLE = "100m_v_component_of_wind"

HORIZON = 360


# -------------------------
# Parse cycle
# -------------------------

cycle = datetime.strptime(
    BUNDLE.name,
    "%Y%m%dT%H%M%SZ"
).replace(
    tzinfo=timezone.utc
)


OUT.mkdir(
    parents=True,
    exist_ok=True,
)


print(
    "Processing",
    BUNDLE.name
)


# -------------------------
# Load GFS v100
# -------------------------

gfs = load_gfs_artifact(
    str(BUNDLE),
    VARIABLE,
    HORIZON,
).float()


print(
    "V100 GFS:",
    gfs.shape,
    gfs.dtype,
)


# -------------------------
# Save
# -------------------------

torch.save(
    gfs.half(),
    OUT / "v100_gfs_holdout_20250201T000000Z.pt",
)


print("Saved:")
print(
    OUT / "v100_gfs_holdout_20250201T000000Z.pt"
)
