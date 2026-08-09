from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import torch
import pandas as pd

from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.features.era5_files import find_era5_files
from evaluation.truth import Era5TruthLoader


BUNDLE_ROOT = "/Zeus/data/evaluation/forecast_store_hist/bundles"
ERA5_ROOT = "/Zeus/data/evaluation/era5"

VARIABLE = "100m_u_component_of_wind"
HORIZON = 360

OUTPUT = "data/evaluation/training/u100_train_all.parquet"


def parse_cycle(path):
    name = Path(path).name
    return datetime.strptime(
        name,
        "%Y%m%dT%H%M%SZ"
    ).replace(tzinfo=timezone.utc)


def sample_cycle(bundle):

    cycle_time = parse_cycle(bundle)

    print("Processing:", cycle_time)

    gfs = load_gfs_artifact(
        bundle,
        VARIABLE,
        HORIZON,
    ).float()

    era5_files = find_era5_files(
        ERA5_ROOT,
        VARIABLE,
        cycle_time,
        HORIZON,
    )

    truth = Era5TruthLoader().load(
        era5_files,
        variable=VARIABLE,
        cycle_time=cycle_time,
        horizon_hours=HORIZON,
    )

    residual = truth.tensor - gfs

    n = 100000

    t = torch.randint(
        0,
        residual.shape[0],
        (n,)
    )

    y = torch.randint(
        0,
        residual.shape[1],
        (n,)
    )

    x = torch.randint(
        0,
        residual.shape[2],
        (n,)
    )


    df = pd.DataFrame(
        {
            "gfs_value": gfs[t,y,x].numpy(),
            "residual": residual[t,y,x].numpy(),
            "lead_hour": t.numpy(),
            "latitude": y.numpy(),
            "longitude": x.numpy(),
        }
    )

    return df



def main():

    bundles = sorted(
        [
            p for p in Path(BUNDLE_ROOT).iterdir()
            if p.is_dir()
        ]
    )

    frames = []

    for bundle in bundles:
        frames.append(
            sample_cycle(bundle)
        )

    train = pd.concat(
        frames,
        ignore_index=True
    )

    Path(OUTPUT).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_parquet(
        OUTPUT
    )

    print(train.head())
    print(train.describe())
    print("saved:", OUTPUT)



if __name__ == "__main__":
    main()
