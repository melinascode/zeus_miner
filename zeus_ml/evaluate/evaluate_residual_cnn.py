from pathlib import Path

import torch
import numpy as np

from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.features.era5_files import find_era5_files
from evaluation.truth import Era5TruthLoader

from zeus_ml.models.residual_cnn import ZeusResidualCNN

from datetime import datetime, timezone
# -------------------------------------------------
# Configuration
# -------------------------------------------------

BUNDLE = Path(
    "/Zeus/data/evaluation/forecast_store_hist/bundles/"
    "20260709T000000Z"
)

ERA5_PATH = (
    "/Zeus/data/evaluation/era5"
)

MODEL_PATH = (
    "/Zeus/data/evaluation/training/"
    "zeus_residual_cnn_12cycles.pt"
)

WEIGHT_PATH = (
    "/Zeus/zeus/data/weights/"
    "latitude_weights_for_rmse.npy"
)


HORIZON = 360


VARIABLES = [
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
]


NAMES = [
    "temperature",
    "u100",
    "v100",
    "ssrd",
]


# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# -------------------------------------------------
# Load latitude weights
# -------------------------------------------------

weights = np.load(
    WEIGHT_PATH
)

weights = torch.tensor(
    weights,
    dtype=torch.float32
)

print(
    "Latitude weights:",
    weights.shape
)


# -------------------------------------------------
# Load GFS + ERA5
# -------------------------------------------------

print(
    "Loading bundle:",
    BUNDLE.name
)

cycle = datetime.strptime(
    BUNDLE.name,
    "%Y%m%dT%H%M%SZ"
).replace(
    tzinfo=timezone.utc
)

gfs_channels = []
era5_channels = []


for variable in VARIABLES:

    print(
        "Loading:",
        variable
    )


    gfs = load_gfs_artifact(
        str(BUNDLE),
        variable,
        HORIZON,
    ).float()


    files = find_era5_files(
        ERA5_PATH,
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


    gfs_channels.append(
        gfs
    )

    era5_channels.append(
        era5
    )


gfs = torch.stack(
    gfs_channels,
    dim=0
)


era5 = torch.stack(
    era5_channels,
    dim=0
)


print(
    "GFS:",
    gfs.shape
)

print(
    "ERA5:",
    era5.shape
)


# -------------------------------------------------
# Residual CNN
# -------------------------------------------------

model = ZeusResidualCNN(
    in_channels=4,
    out_channels=4,
    hidden_channels=16,
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)


model = model.to(
    device
)

model.eval()


with torch.no_grad():

    predicted_residual = torch.zeros_like(
        gfs
    )

    for hour in range(
        HORIZON + 1
    ):

        x = gfs[:, hour]

        x = x.unsqueeze(
            0
        )


        pred = model(
            x.to(device)
        )


        predicted_residual[:, hour] = (
            pred
            .squeeze(0)
            .cpu()
        )


        if hour % 50 == 0:
            print(
                "Predicted hour:",
                hour
            )


cnn_corrected = (
    gfs +
    predicted_residual
)


print(
    "CNN corrected:",
    cnn_corrected.shape
)


# -------------------------------------------------
# iwRMSE
# -------------------------------------------------

def iwRMSE(
    prediction,
    truth,
    latitude_weights,
):

    error = (
        prediction -
        truth
    ) ** 2


    weighted = (
        error *
        latitude_weights[None, :, None]
    )


    return torch.sqrt(
        weighted.sum()
        /
        latitude_weights.sum()
        /
        error.shape[-1]
        /
        error.shape[0]
    )


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

print("\n==============================")
print("Latitude weighted iwRMSE")
print("==============================")


for i, name in enumerate(NAMES):


    raw_score = iwRMSE(
        gfs[i],
        era5[i],
        weights,
    )


    cnn_score = iwRMSE(
        cnn_corrected[i],
        era5[i],
        weights,
    )


    improvement = (
        (
            raw_score -
            cnn_score
        )
        /
        raw_score
        *
        100
    )


    print()

    print(
        "====",
        name,
        "===="
    )


    print(
        "Raw iwRMSE:",
        float(raw_score)
    )


    print(
        "CNN iwRMSE:",
        float(cnn_score)
    )


    print(
        "Improvement %:",
        float(improvement)
    )