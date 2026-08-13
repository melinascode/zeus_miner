from pathlib import Path
from datetime import datetime, timezone

import torch

from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.features.era5_files import find_era5_files
from evaluation.truth import Era5TruthLoader


BUNDLES = Path(
    "/Zeus/data/evaluation/forecast_store_hist/bundles"
)

ERA5 = "/Zeus/data/evaluation/era5"

OUT = Path(
    "/Zeus/data/evaluation/training/cnn_residual"
)


VARIABLES = [
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
]


HORIZON = 360


OUT.mkdir(
    parents=True,
    exist_ok=True,
)


bundles = sorted(
    [
        b for b in BUNDLES.iterdir()
        if b.is_dir()
    ]
)


# same split as previous ML work
train_bundles = bundles[2:]


for bundle in train_bundles:

    print(
        "Processing:",
        bundle.name
    )


    cycle = datetime.strptime(
        bundle.name,
        "%Y%m%dT%H%M%SZ"
    ).replace(
        tzinfo=timezone.utc
    )


    gfs_fields = []
    residual_fields = []


    for variable in VARIABLES:

        print(
            "Loading:",
            variable
        )


        gfs = load_gfs_artifact(
            str(bundle),
            variable,
            HORIZON,
        ).float()


        era5_files = find_era5_files(
            ERA5,
            variable,
            cycle,
            HORIZON,
        )


        truth = Era5TruthLoader().load(
            era5_files,
            variable=variable,
            cycle_time=cycle,
            horizon_hours=HORIZON,
        )


        era5 = truth.tensor.float()


        if era5.shape != gfs.shape:
            raise RuntimeError(
                f"Shape mismatch {variable}: "
                f"{era5.shape} vs {gfs.shape}"
            )


        residual = (
            era5 -
            gfs
        )


        gfs_fields.append(gfs)
        residual_fields.append(residual)



    # (lead, channel, lat, lon)
    gfs_tensor = torch.stack(
        gfs_fields,
        dim=1,
    )


    residual_tensor = torch.stack(
        residual_fields,
        dim=1,
    )


    print(
        "Input:",
        gfs_tensor.shape
    )

    print(
        "Target:",
        residual_tensor.shape
    )


    cycle_dir = OUT / bundle.name

    cycle_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # save one lead hour = one CNN sample
    for lead in range(
        gfs_tensor.shape[0]
    ):

        torch.save(
            gfs_tensor[lead].half(),
            cycle_dir /
            f"input_{lead:03d}.pt",
        )


        torch.save(
            residual_tensor[lead].half(),
            cycle_dir /
            f"target_{lead:03d}.pt",
        )


    del gfs_tensor
    del residual_tensor


print(
    "CNN dataset finished"
)