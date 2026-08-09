import pandas as pd
import xgboost as xgb
import numpy as np


DATA = "data/evaluation/training/u100_train_uv_spatial_5M.parquet"

OUT = "data/evaluation/training/u100_xgb_uv_spatial.json"


# -------------------------
# Load data
# -------------------------

df = pd.read_parquet(DATA)

print("Loaded:", df.shape)


# -------------------------
# Reduce training data
# -------------------------

df = df.sample(
    n=30_000_000,
    random_state=42
)

print("Sampled:", df.shape)


# -------------------------
# Feature engineering
# -------------------------

df["gfs_abs"] = np.abs(
    df["gfs_value"]
)

df["gfs_squared"] = (
    df["gfs_value"] ** 2
)

df["lead_day"] = (
    df["lead_hour"] / 24.0
)

df["abs_latitude"] = np.abs(
    df["latitude"]
)


lat_rad = np.deg2rad(
    df["latitude"]
)

lon_rad = np.deg2rad(
    df["longitude"]
)


df["lat_sin"] = np.sin(
    lat_rad
)

df["lat_cos"] = np.cos(
    lat_rad
)

df["lon_sin"] = np.sin(
    lon_rad
)

df["lon_cos"] = np.cos(
    lon_rad
)


# -------------------------
# Features
# -------------------------

features = [

    # u100 spatial
    "gfs_value",
    "gfs_north",
    "gfs_south",
    "gfs_east",
    "gfs_west",
    "gradient_lat",
    "gradient_lon",

    # v100 spatial
    "v100_value",
    "v100_north",
    "v100_south",
    "v100_east",
    "v100_west",
    "v100_gradient_lat",
    "v100_gradient_lon",

    # time/location
    "lead_hour",
    "latitude",
    "longitude",

    # engineered
    "gfs_abs",
    "gfs_squared",
    "lead_day",
    "abs_latitude",
    "lat_sin",
    "lat_cos",
    "lon_sin",
    "lon_cos",
]


X = df[features]

y = df["residual"]


print(
    "Features:",
    X.shape
)


# -------------------------
# GPU Model
# -------------------------

model = xgb.XGBRegressor(

    n_estimators=3000,

    max_depth=10,

    learning_rate=0.015,

    subsample=0.9,

    colsample_bytree=0.8,

    reg_alpha=0.1,

    reg_lambda=5,

    tree_method="hist",

    device="cuda",

    max_bin=64,

    n_jobs=-1,
)


# -------------------------
# Train
# -------------------------

model.fit(
    X,
    y,
)


# -------------------------
# Save
# -------------------------

model.save_model(
    OUT
)


print(
    "saved:",
    OUT
)