from pathlib import Path
import torch

from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.features.era5_files import find_era5_files
from evaluation.truth import Era5TruthLoader

from datetime import datetime, timezone


BUNDLES = Path(
    "/Zeus/data/evaluation/forecast_store_hist/bundles"
)

ERA5 = "/Zeus/data/evaluation/era5"

OUT = Path(
    "/Zeus/data/evaluation/residual_ml/cycles"
)

VARIABLE = "100m_u_component_of_wind"
HORIZON = 360


def parse_cycle(p):
    return datetime.strptime(
        p.name,
        "%Y%m%dT%H%M%SZ"
    ).replace(
        tzinfo=timezone.utc
    )


OUT.mkdir(
    parents=True,
    exist_ok=True
)


for bundle in sorted(BUNDLES.iterdir()):

    if not bundle.is_dir():
        continue

    cycle = parse_cycle(bundle)

    name = bundle.name

    print("Processing", name)


    gfs = load_gfs_artifact(
        str(bundle),
        VARIABLE,
        HORIZON,
    ).float()


    era5_files = find_era5_files(
        ERA5,
        VARIABLE,
        cycle,
        HORIZON,
    )


    truth = Era5TruthLoader().load(
        era5_files,
        variable=VARIABLE,
        cycle_time=cycle,
        horizon_hours=HORIZON,
    )


    residual = (
        truth.tensor -
        gfs
    )


    torch.save(
        gfs.half(),
        OUT / f"{name}_gfs.pt"
    )

    torch.save(
        residual.float(),
        OUT / f"{name}_residual.pt"
    )


    print(
        "saved",
        name,
        residual.shape
    )
