from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import torch

from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.features.era5_files import find_era5_files
from evaluation.truth import Era5TruthLoader


def build_single_residual():

    # Zeus cycle
    cycle_time = datetime(
        2025,
        4,
        22,
        18,
        tzinfo=timezone.utc,
    )

    horizon = 360

    variable = "100m_u_component_of_wind"

    # -------------------------
    # Load GFS
    # -------------------------

    bundle = (
        "/Zeus/data/evaluation/"
        "forecast_store_hist/bundles/"
        "20250422T180000Z"
    )

    gfs = load_gfs_artifact(
        bundle,
        variable,
        horizon,
    )

    print(
        "GFS:",
        gfs.shape,
        gfs.dtype,
    )


    # -------------------------
    # Find ERA5 files
    # -------------------------

    era5_files = find_era5_files(
        "/Zeus/data/evaluation/era5",
        variable,
        cycle_time,
        horizon,
    )

    print(
        "ERA5 files:",
        len(era5_files),
    )


    # -------------------------
    # Load ERA5 truth
    # -------------------------

    loader = Era5TruthLoader()

    truth = loader.load(
        era5_files,
        variable=variable,
        cycle_time=cycle_time,
        horizon_hours=horizon,
    )

    era5 = truth.tensor

    print(
        "ERA5:",
        era5.shape,
        era5.dtype,
    )


    # -------------------------
    # Residual
    # -------------------------

    residual = (
        era5 -
        gfs.float()
    )

    print(
        "Residual:",
        residual.shape,
        residual.dtype,
    )

    print(
        "Residual mean:",
        residual.mean().item()
    )

    print(
        "Residual std:",
        residual.std().item()
    )


if __name__ == "__main__":
    build_single_residual()