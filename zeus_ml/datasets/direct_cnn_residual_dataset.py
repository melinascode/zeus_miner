from pathlib import Path
from datetime import datetime, timezone

import torch
from torch.utils.data import Dataset

from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.features.era5_files import find_era5_files
from evaluation.truth import Era5TruthLoader


VARIABLES = [
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
]


class DirectCNNResidualDataset(Dataset):

    def __init__(
        self,
        bundles,
        era5_path,
        horizon=360,
    ):

        self.samples = []

        for bundle in bundles:

            bundle = Path(bundle)

            for hour in range(horizon + 1):

                self.samples.append(
                    (
                        bundle,
                        hour,
                    )
                )


        self.era5_path = era5_path
        self.horizon = horizon


    def __len__(self):

        return len(self.samples)


    def __getitem__(
        self,
        idx,
    ):

        bundle, hour = self.samples[idx]


        print(
            "Loading:",
            bundle.name,
            "hour:",
            hour,
        )


        cycle = datetime.strptime(
            bundle.name,
            "%Y%m%dT%H%M%SZ"
        ).replace(
            tzinfo=timezone.utc
        )


        gfs_channels = []
        era5_channels = []


        for variable in VARIABLES:

            gfs = load_gfs_artifact(
                str(bundle),
                variable,
                self.horizon,
            ).float()


            files = find_era5_files(
                self.era5_path,
                variable,
                cycle,
                self.horizon,
            )


            truth = Era5TruthLoader().load(
                files,
                variable=variable,
                cycle_time=cycle,
                horizon_hours=self.horizon,
            )


            era5 = truth.tensor.float()


            gfs_channels.append(
                gfs[hour]
            )

            era5_channels.append(
                era5[hour]
            )


        gfs = torch.stack(
            gfs_channels,
            dim=0,
        )


        era5 = torch.stack(
            era5_channels,
            dim=0,
        )


        residual = (
            era5 -
            gfs
        )


        return (
            gfs,
            residual,
        )