from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.features.era5_files import find_era5_files
from evaluation.truth import Era5TruthLoader


BUNDLES = Path(
    "/Zeus/data/evaluation/forecast_store_hist/bundles"
)

ERA5 = "/Zeus/data/evaluation/era5"

OUT = Path(
    "/Zeus/data/evaluation/training/u100_train_multi_spatial_5M.parquet"
)


VARIABLE = "100m_u_component_of_wind"

HORIZON = 360

SAMPLES_PER_CYCLE = 5_000_000


# -------------------------
# Split bundles
# -------------------------

bundles = sorted(
    [
        b for b in BUNDLES.iterdir()
        if b.is_dir()
    ]
)


test_bundles = bundles[:2]

train_bundles = bundles[2:]


print(
    "Total bundles:",
    len(bundles)
)

print(
    "Holdout bundles:"
)

for b in test_bundles:
    print(
        b.name
    )


print(
    "Training bundles:",
    len(train_bundles)
)


rows = []


# -------------------------
# Build training data
# -------------------------

for bundle in train_bundles:

    name = bundle.name

    print(
        "Processing",
        name
    )


    cycle = datetime.strptime(
        name,
        "%Y%m%dT%H%M%SZ"
    ).replace(
        tzinfo=timezone.utc
    )


    # -------------------------
    # Load GFS
    # -------------------------

    gfs = load_gfs_artifact(
        str(bundle),
        VARIABLE,
        HORIZON,
    ).float()


    # -------------------------
    # Load ERA5
    # -------------------------

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


    hours, lat_size, lon_size = gfs.shape


    # -------------------------
    # Spatial features
    # -------------------------

    gfs_np = gfs.numpy()

    residual_np = residual.numpy()


    north = np.empty_like(gfs_np)
    south = np.empty_like(gfs_np)
    east = np.empty_like(gfs_np)
    west = np.empty_like(gfs_np)


    # latitude neighbors

    north[:, :-1, :] = gfs_np[:, 1:, :]
    north[:, -1, :] = gfs_np[:, -1, :]


    south[:, 1:, :] = gfs_np[:, :-1, :]
    south[:, 0, :] = gfs_np[:, 0, :]


    # longitude wrap

    east[:, :, :-1] = gfs_np[:, :, 1:]
    east[:, :, -1] = gfs_np[:, :, 0]


    west[:, :, 1:] = gfs_np[:, :, :-1]
    west[:, :, 0] = gfs_np[:, :, -1]


    gradient_lat = (
        north -
        south
    )


    gradient_lon = (
        east -
        west
    )


    # -------------------------
    # Random sampling
    # -------------------------

    total = (
        hours *
        lat_size *
        lon_size
    )


    indices = np.random.choice(
        total,
        size=SAMPLES_PER_CYCLE,
        replace=False,
    )


    pixels = (
        lat_size *
        lon_size
    )


    lead = indices // pixels

    rem = indices % pixels

    lat_idx = rem // lon_size

    lon_idx = rem % lon_size


    latitude = np.linspace(
        -90,
        90,
        lat_size,
    )


    longitude = np.linspace(
        -180,
        180,
        lon_size,
        endpoint=False,
    )


    flat_gfs = gfs_np.reshape(-1)

    flat_residual = residual_np.reshape(-1)

    flat_north = north.reshape(-1)

    flat_south = south.reshape(-1)

    flat_east = east.reshape(-1)

    flat_west = west.reshape(-1)

    flat_grad_lat = gradient_lat.reshape(-1)

    flat_grad_lon = gradient_lon.reshape(-1)


    df = pd.DataFrame(
        {
            "gfs_value":
                flat_gfs[indices],

            "gfs_north":
                flat_north[indices],

            "gfs_south":
                flat_south[indices],

            "gfs_east":
                flat_east[indices],

            "gfs_west":
                flat_west[indices],

            "gradient_lat":
                flat_grad_lat[indices],

            "gradient_lon":
                flat_grad_lon[indices],

            "residual":
                flat_residual[indices],

            "lead_hour":
                lead,

            "latitude":
                latitude[lat_idx],

            "longitude":
                longitude[lon_idx],

            "cycle_id":
                name,
        }
    )


    rows.append(df)


    print(
        "samples:",
        len(df)
    )


# -------------------------
# Save
# -------------------------

final = pd.concat(
    rows,
    ignore_index=True,
)


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


final.to_parquet(
    OUT,
    index=False,
)


print(
    "Shape:",
    final.shape
)

print(
    "Cycles:",
    final.cycle_id.nunique()
)

print(
    final.head()
)

print(
    "Saved:",
    OUT
)