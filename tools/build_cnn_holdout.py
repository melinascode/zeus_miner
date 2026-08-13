from pathlib import Path
from datetime import datetime, timezone

import torch

from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.features.era5_files import find_era5_files
from evaluation.truth import Era5TruthLoader


BUNDLE = Path(
    "/Zeus/data/evaluation/forecast_store_hist/bundles/20250201T000000Z"
)

ERA5 = "/Zeus/data/evaluation/era5"

OUT = Path(
    "/Zeus/data/evaluation/residual_ml/holdout"
)

VARIABLES = [
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
]

HORIZON = 360


cycle = datetime.strptime(
    BUNDLE.name,
    "%Y%m%dT%H%M%SZ"
).replace(
    tzinfo=timezone.utc
)


gfs_channels = []
era5_channels = []


for variable in VARIABLES:

    print("Loading:", variable)

    gfs = load_gfs_artifact(
        str(BUNDLE),
        variable,
        HORIZON,
    ).float()


    files = find_era5_files(
        ERA5,
        variable,
        cycle,
        HORIZON,
    )


    truth = Era5TruthLoader().load(
        files,
        variable=variable,
        cycle_time=cycle,
        horizon_hours=HORIZON,
    )


    era5 = truth.tensor.float()


    gfs_channels.append(gfs)
    era5_channels.append(era5)



gfs = torch.stack(
    gfs_channels,
    dim=1,
)

era5 = torch.stack(
    era5_channels,
    dim=1,
)


residual = (
    era5 - gfs
)


OUT.mkdir(
    parents=True,
    exist_ok=True,
)


torch.save(
    gfs.half(),
    OUT / "cnn_gfs_holdout.pt",
)


torch.save(
    residual.float(),
    OUT / "cnn_residual_holdout.pt",
)


print("GFS:", gfs.shape)
print("Residual:", residual.shape)
