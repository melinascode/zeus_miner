from pathlib import Path

import numpy as np
import pandas as pd
import torch


def build_features(
    gfs_path,
    residual_path,
    output_path,
    sample_size=1000000,
    random_seed=42,
):

    print("Loading tensors...")

    gfs = torch.load(gfs_path).float()
    residual = torch.load(residual_path).float()

    print("GFS:", gfs.shape)
    print("Residual:", residual.shape)

    assert gfs.shape == residual.shape


    hours, lat_size, lon_size = gfs.shape


    print("Creating coordinates...")

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


    print("Sampling points...")

    total = hours * lat_size * lon_size

    rng = np.random.default_rng(
        random_seed
    )

    indices = rng.choice(
        total,
        size=sample_size,
        replace=False,
    )


    # -------------------------
    # Decode tensor indices
    # -------------------------

    lead = indices // (
        lat_size * lon_size
    )

    remainder = indices % (
        lat_size * lon_size
    )

    lat_idx = remainder // lon_size

    lon_idx = remainder % lon_size


    # -------------------------
    # Raw values
    # -------------------------

    gfs_values = (
        gfs.numpy()
        .reshape(-1)[indices]
    )

    residual_values = (
        residual.numpy()
        .reshape(-1)[indices]
    )


    lat_values = latitude[lat_idx]

    lon_values = longitude[lon_idx]


    # -------------------------
    # Feature engineering
    # -------------------------

    lat_rad = np.deg2rad(
        lat_values
    )

    lon_rad = np.deg2rad(
        lon_values
    )


    df = pd.DataFrame(
        {
            # original features

            "gfs_value":
                gfs_values,

            "lead_hour":
                lead,

            "latitude":
                lat_values,

            "longitude":
                lon_values,


            # nonlinear wind features

            "gfs_abs":
                np.abs(gfs_values),

            "gfs_squared":
                gfs_values ** 2,


            # forecast age

            "lead_day":
                lead / 24.0,


            # latitude structure

            "abs_latitude":
                np.abs(lat_values),

            "lat_sin":
                np.sin(lat_rad),

            "lat_cos":
                np.cos(lat_rad),


            # longitude periodicity

            "lon_sin":
                np.sin(lon_rad),

            "lon_cos":
                np.cos(lon_rad),


            # target

            "residual":
                residual_values,
        }
    )


    print(df.head())


    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    df.to_parquet(
        output_path,
        index=False,
    )


    print(
        "Saved:",
        output_path,
    )

    print(df.describe())


if __name__ == "__main__":

    build_features(
        gfs_path=
        "data/evaluation/residual_ml/u100_gfs.pt",

        residual_path=
        "data/evaluation/residual_ml/u100_residual_20250422T180000Z.pt",

        output_path=
        "data/evaluation/training/u100_train.parquet",

        sample_size=1000000,
    )