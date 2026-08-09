import torch
import pandas as pd
import xgboost as xgb
import numpy as np


# -------------------------
# Load tensors
# -------------------------

u100_gfs = torch.load(
    "data/evaluation/residual_ml/holdout/u100_gfs_holdout_20250201T000000Z.pt"
).float()


v100_gfs = torch.load(
    "data/evaluation/residual_ml/holdout/v100_gfs_holdout_20250201T000000Z.pt"
).float()


residual = torch.load(
    "data/evaluation/residual_ml/holdout/u100_residual_holdout_20250201T000000Z.pt"
).float()


print("U100 GFS:", u100_gfs.shape)
print("V100 GFS:", v100_gfs.shape)
print("Residual:", residual.shape)



# -------------------------
# Load model
# -------------------------

model = xgb.XGBRegressor()

model.load_model(
    "data/evaluation/training/u100_xgb_uv_spatial.json"
)



# -------------------------
# Build features
# -------------------------

hours, lat_size, lon_size = u100_gfs.shape


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



u_np = u100_gfs.numpy()

v_np = v100_gfs.numpy()



chunk_size = 1_000_000


predictions = []


flat_u = u_np.reshape(-1)


pixels = lat_size * lon_size


total = len(flat_u)



for i in range(0, total, chunk_size):

    print(
        "Predicting",
        i,
        "/",
        total
    )


    end = min(
        i + chunk_size,
        total
    )


    idx = np.arange(
        i,
        end
    )


    lead = idx // pixels

    rem = idx % pixels

    lat_idx = rem // lon_size

    lon_idx = rem % lon_size



    # -------------------------
    # u100 spatial
    # -------------------------

    north_idx = np.clip(
        lat_idx + 1,
        0,
        lat_size - 1
    )

    south_idx = np.clip(
        lat_idx - 1,
        0,
        lat_size - 1
    )


    east_idx = (
        lon_idx + 1
    ) % lon_size


    west_idx = (
        lon_idx - 1
    ) % lon_size



    gfs_value = u_np[
        lead,
        lat_idx,
        lon_idx
    ]


    gfs_north = u_np[
        lead,
        north_idx,
        lon_idx
    ]

    gfs_south = u_np[
        lead,
        south_idx,
        lon_idx
    ]

    gfs_east = u_np[
        lead,
        lat_idx,
        east_idx
    ]

    gfs_west = u_np[
        lead,
        lat_idx,
        west_idx
    ]


    gradient_lat = (
        gfs_north -
        gfs_south
    )


    gradient_lon = (
        gfs_east -
        gfs_west
    )



    # -------------------------
    # v100 spatial
    # -------------------------

    v100_value = v_np[
        lead,
        lat_idx,
        lon_idx
    ]


    v100_north = v_np[
        lead,
        north_idx,
        lon_idx
    ]


    v100_south = v_np[
        lead,
        south_idx,
        lon_idx
    ]


    v100_east = v_np[
        lead,
        lat_idx,
        east_idx
    ]


    v100_west = v_np[
        lead,
        lat_idx,
        west_idx
    ]


    v100_gradient_lat = (
        v100_north -
        v100_south
    )


    v100_gradient_lon = (
        v100_east -
        v100_west
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


    chunk = pd.DataFrame(
        {

            "gfs_value":
                gfs_value,

            "gfs_north":
                gfs_north,

            "gfs_south":
                gfs_south,

            "gfs_east":
                gfs_east,

            "gfs_west":
                gfs_west,

            "gradient_lat":
                gradient_lat,

            "gradient_lon":
                gradient_lon,


            "v100_value":
                v100_value,

            "v100_north":
                v100_north,

            "v100_south":
                v100_south,

            "v100_east":
                v100_east,

            "v100_west":
                v100_west,

            "v100_gradient_lat":
                v100_gradient_lat,

            "v100_gradient_lon":
                v100_gradient_lon,


            "lead_hour":
                lead,

            "latitude":
                lat_values,

            "longitude":
                lon_values,


            "gfs_abs":
                np.abs(gfs_value),

            "gfs_squared":
                gfs_value ** 2,

            "lead_day":
                lead / 24.0,

            "abs_latitude":
                np.abs(lat_values),

            "lat_sin":
                np.sin(lat_rad),

            "lat_cos":
                np.cos(lat_rad),

            "lon_sin":
                np.sin(lon_rad),

            "lon_cos":
                np.cos(lon_rad),
        }
    )


    predictions.append(
        model.predict(chunk)
    )



# -------------------------
# Combine prediction
# -------------------------

predicted_residual = np.concatenate(
    predictions
)


predicted_residual = torch.tensor(
    predicted_residual,
    dtype=torch.float32
).reshape(
    u100_gfs.shape
)



# -------------------------
# Error
# -------------------------

corrected_error = (
    residual -
    predicted_residual
)



# -------------------------
# Plain RMSE
# -------------------------

raw_rmse = torch.sqrt(
    torch.mean(
        residual ** 2
    )
)


corrected_rmse = torch.sqrt(
    torch.mean(
        corrected_error ** 2
    )
)



# -------------------------
# Zeus latitude weights
# -------------------------

weights = np.load(
    "zeus/data/weights/latitude_weights_for_rmse.npy"
)


weights = torch.tensor(
    weights,
    dtype=torch.float32
)


weights = weights.view(
    1,
    -1,
    1
)


weights = (
    weights /
    weights.mean()
)



def zeus_weighted_rmse(
    error,
    weights
):

    return torch.sqrt(
        torch.mean(
            error ** 2 *
            weights
        )
    )



def zeus_weighted_mae(
    error,
    weights
):

    return torch.mean(
        torch.abs(error) *
        weights
    )



raw_iwrmse = zeus_weighted_rmse(
    residual,
    weights
)


corrected_iwrmse = zeus_weighted_rmse(
    corrected_error,
    weights
)



raw_iwmae = zeus_weighted_mae(
    residual,
    weights
)


corrected_iwmae = zeus_weighted_mae(
    corrected_error,
    weights
)



# -------------------------
# Results
# -------------------------

print()

print("===== Plain RMSE =====")

print(
    "Raw GFS RMSE:",
    raw_rmse.item()
)

print(
    "Corrected RMSE:",
    corrected_rmse.item()
)

print(
    "Improvement %:",
    (
        1 -
        corrected_rmse /
        raw_rmse
    ).item() * 100
)



print()

print("===== Zeus Latitude Weighted RMSE =====")

print(
    "Raw GFS iwRMSE:",
    raw_iwrmse.item()
)

print(
    "Corrected iwRMSE:",
    corrected_iwrmse.item()
)

print(
    "Improvement %:",
    (
        1 -
        corrected_iwrmse /
        raw_iwrmse
    ).item() * 100
)



print()

print("===== Zeus Latitude Weighted MAE =====")

print(
    "Raw GFS iwMAE:",
    raw_iwmae.item()
)

print(
    "Corrected iwMAE:",
    corrected_iwmae.item()
)

print(
    "Improvement %:",
    (
        1 -
        corrected_iwmae /
        raw_iwmae
    ).item() * 100
)