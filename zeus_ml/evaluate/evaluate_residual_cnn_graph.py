from pathlib import Path
from datetime import datetime, timezone

import torch
import numpy as np
import matplotlib.pyplot as plt

from zeus_ml.features.gfs_loader import load_gfs_artifact
from zeus_ml.features.era5_files import find_era5_files
from evaluation.truth import Era5TruthLoader

from zeus_ml.models.residual_cnn import ZeusResidualCNN


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

OUTPUT = (
    "/Zeus/data/evaluation/training/"
    "cnn_vs_gfs_20260709T000000Z.png"
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
# Latitude weights
# -------------------------------------------------

weights = torch.tensor(
    np.load(
        WEIGHT_PATH
    ),
    dtype=torch.float32
)


print(
    "Latitude weights:",
    weights.shape
)


# -------------------------------------------------
# Load data
# -------------------------------------------------

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
    gfs_channels
)

era5 = torch.stack(
    era5_channels
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
# Load CNN
# -------------------------------------------------

model = ZeusResidualCNN(
    in_channels=4,
    out_channels=4,
    hidden_channels=16,
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model.to(device)
model.eval()


# -------------------------------------------------
# Hourly prediction
# -------------------------------------------------

predicted_residual = torch.zeros_like(
    gfs
)


with torch.no_grad():

    for hour in range(
        HORIZON + 1
    ):

        x = gfs[:, hour]


        prediction = model(
            x.unsqueeze(0).to(device)
        )


        predicted_residual[:, hour] = (
            prediction
            .squeeze(0)
            .cpu()
        )


        if hour % 50 == 0:
            print(
                "Prediction hour:",
                hour
            )


cnn = (
    gfs +
    predicted_residual
)


# -------------------------------------------------
# Hourly latitude weighted iwRMSE
# -------------------------------------------------

def hourly_iwRMSE(
    prediction,
    truth,
):

    results = []

    for hour in range(
        HORIZON + 1
    ):

        pred_hour = prediction[:, hour]

        truth_hour = truth[:, hour]


        error = (
            pred_hour -
            truth_hour
        ) ** 2


        # error:
        # [channels, latitude, longitude]

        weighted = (
            error *
            weights[None, :, None]
        )


        score = torch.sqrt(
            weighted.sum()
            /
            weights.sum()
            /
            error.shape[-1]
            /
            error.shape[0]
        )


        results.append(
            score
        )


    return torch.stack(
        results
    )
# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(
    figsize=(12,6)
)


hours = np.arange(
    HORIZON + 1
)


for i, name in enumerate(
    NAMES
):

    raw_score = hourly_iwRMSE(
        gfs[i:i+1],
        era5[i:i+1],
    )


    cnn_score = hourly_iwRMSE(
        cnn[i:i+1],
        era5[i:i+1],
    )


    plt.figure(
        figsize=(12,5)
    )


    plt.plot(
        hours,
        raw_score.numpy(),
        label="Raw GFS"
    )


    plt.plot(
        hours,
        cnn_score.numpy(),
        label="CNN corrected"
    )


    plt.xlabel(
        "Forecast hour"
    )

    plt.ylabel(
        "Latitude weighted iwRMSE"
    )


    plt.title(
        f"{name} - 20260709T000000Z"
    )


    plt.legend()

    plt.grid(
        True
    )


    filename = (
        OUTPUT.replace(
            ".png",
            f"_{name}.png"
        )
    )


    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved:",
        filename
    )


print(
    "Finished"
)