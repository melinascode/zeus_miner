from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data = pd.read_parquet(
    "data/evaluation/training/u100_train.parquet"
)


X = data[
    [
        "gfs_value",
        "lead_hour",
        "latitude",
        "longitude",
    ]
]

y = data["residual"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
)


model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
)


model.fit(
    X_train,
    y_train,
)


pred = model.predict(
    X_test
)


mse = mean_squared_error(
    y_test,
    pred,
)

rmse = mse ** 0.5


print(
    "RMSE:",
    rmse,
)


Path(
    "data/evaluation/models"
).mkdir(
    parents=True,
    exist_ok=True,
)


model.save_model(
    "data/evaluation/models/u100_residual_xgb.json"
)


print(
    "Saved model"
)
