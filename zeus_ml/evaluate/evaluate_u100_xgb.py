from pathlib import Path

import torch
import pandas as pd
import xgboost as xgb


# -------------------------
# Load tensors
# -------------------------

gfs = torch.load(
    "data/evaluation/residual_ml/u100_gfs.pt"
).float()

residual = torch.load(
    "data/evaluation/residual_ml/u100_residual_20250422T180000Z.pt"
).float()


print("GFS:", gfs.shape)
print("Residual:", residual.shape)


# -------------------------
# Load model
# -------------------------

model = xgb.XGBRegressor()

model.load_model(
    "data/evaluation/models/u100_residual_xgb.json"
)


# -------------------------
# Build evaluation features
# -------------------------

hours, lat_size, lon_size = gfs.shape


latitude = torch.linspace(
    -90,
    90,
    lat_size,
)

longitude = torch.linspace(
    -180,
    180,
    lon_size + 1,
)[:-1]


total = hours * lat_size * lon_size


df = pd.DataFrame(
    {
        "gfs_value":
            gfs.reshape(-1).numpy(),

        "lead_hour":
            torch.arange(hours)
            .repeat_interleave(lat_size * lon_size)
            .numpy(),

        "latitude":
            latitude.repeat_interleave(
                lon_size
            ).repeat(hours).numpy(),

        "longitude":
            longitude.repeat(
                lat_size
            ).repeat(hours).numpy(),
    }
)


# -------------------------
# Predict residual
# -------------------------

predicted_residual = model.predict(df)


predicted_residual = torch.tensor(
    predicted_residual
).reshape(
    gfs.shape
)


# -------------------------
# Correct GFS
# -------------------------

corrected = (
    gfs +
    predicted_residual
)


# -------------------------
# RMSE
# -------------------------

raw_rmse = torch.sqrt(
    torch.mean(
        residual ** 2
    )
)


corrected_error = (
    residual -
    predicted_residual
)


corrected_rmse = torch.sqrt(
    torch.mean(
        corrected_error ** 2
    )
)


improvement = (
    1 -
    corrected_rmse / raw_rmse
) * 100


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
    improvement.item()
)
