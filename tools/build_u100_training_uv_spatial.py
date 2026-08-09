from pathlib import Path
from datetime import datetime, timezone
import gc

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.features.era5_files import find_era5_files
from evaluation.truth import Era5TruthLoader


BUNDLES = Path(
    "/Zeus/data/evaluation/forecast_store_hist/bundles"
)

ERA5 = "/Zeus/data/evaluation/era5"

OUT = Path(
    "/Zeus/data/evaluation/training/u100_train_uv_spatial_5M.parquet"
)


VARIABLE_U = "100m_u_component_of_wind"
VARIABLE_V = "100m_v_component_of_wind"

HORIZON = 360

SAMPLES_PER_CYCLE = 5_000_000


def make_spatial(x):

    north = np.empty_like(x)
    south = np.empty_like(x)
    east = np.empty_like(x)
    west = np.empty_like(x)

    north[:, :-1, :] = x[:, 1:, :]
    north[:, -1, :] = x[:, -1, :]

    south[:, 1:, :] = x[:, :-1, :]
    south[:, 0, :] = x[:, 0, :]

    east[:, :, :-1] = x[:, :, 1:]
    east[:, :, -1] = x[:, :, 0]

    west[:, :, 1:] = x[:, :, :-1]
    west[:, :, 0] = x[:, :, -1]

    return (
        north,
        south,
        east,
        west,
        north - south,
        east - west,
    )


bundles = sorted(
    [
        b for b in BUNDLES.iterdir()
        if b.is_dir()
    ]
)

test_bundles = bundles[:2]
train_bundles = bundles[2:]


print("Total bundles:", len(bundles))

print("Holdout bundles:")
for b in test_bundles:
    print(b.name)

print("Training bundles:", len(train_bundles))


if OUT.exists():
    OUT.unlink()


writer = None


for bundle in train_bundles:

    name = bundle.name

    print("Processing", name)


    cycle = datetime.strptime(
        name,
        "%Y%m%dT%H%M%SZ"
    ).replace(
        tzinfo=timezone.utc
    )


    gfs_u = load_gfs_artifact(
        str(bundle),
        VARIABLE_U,
        HORIZON,
    ).float()


    gfs_v = load_gfs_artifact(
        str(bundle),
        VARIABLE_V,
        HORIZON,
    ).float()


    era5_files = find_era5_files(
        ERA5,
        VARIABLE_U,
        cycle,
        HORIZON,
    )


    truth = Era5TruthLoader().load(
        era5_files,
        variable=VARIABLE_U,
        cycle_time=cycle,
        horizon_hours=HORIZON,
    )


    residual = truth.tensor - gfs_u


    u = gfs_u.numpy()
    v = gfs_v.numpy()
    r = residual.numpy()


    (
        u_north,
        u_south,
        u_east,
        u_west,
        u_grad_lat,
        u_grad_lon,
    ) = make_spatial(u)


    (
        v_north,
        v_south,
        v_east,
        v_west,
        v_grad_lat,
        v_grad_lon,
    ) = make_spatial(v)


    hours, lat_size, lon_size = u.shape


    total = hours * lat_size * lon_size


    indices = np.random.choice(
        total,
        size=SAMPLES_PER_CYCLE,
        replace=False,
    )


    pixels = lat_size * lon_size


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


    df = pd.DataFrame(
        {

            "gfs_value":
                u.reshape(-1)[indices],

            "gfs_north":
                u_north.reshape(-1)[indices],

            "gfs_south":
                u_south.reshape(-1)[indices],

            "gfs_east":
                u_east.reshape(-1)[indices],

            "gfs_west":
                u_west.reshape(-1)[indices],

            "gradient_lat":
                u_grad_lat.reshape(-1)[indices],

            "gradient_lon":
                u_grad_lon.reshape(-1)[indices],


            "v100_value":
                v.reshape(-1)[indices],

            "v100_north":
                v_north.reshape(-1)[indices],

            "v100_south":
                v_south.reshape(-1)[indices],

            "v100_east":
                v_east.reshape(-1)[indices],

            "v100_west":
                v_west.reshape(-1)[indices],

            "v100_gradient_lat":
                v_grad_lat.reshape(-1)[indices],

            "v100_gradient_lon":
                v_grad_lon.reshape(-1)[indices],


            "residual":
                r.reshape(-1)[indices],

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


    print("samples:", len(df))


    table = pa.Table.from_pandas(df)


    if writer is None:
        writer = pq.ParquetWriter(
            OUT,
            table.schema
        )


    writer.write_table(table)


    del table
    del df

    del gfs_u
    del gfs_v
    del residual

    del u
    del v
    del r

    gc.collect()


if writer is not None:
    writer.close()


print("Saved:", OUT)