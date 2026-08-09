from pathlib import Path

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


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


model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    n_jobs=-1,
)


model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_test, y_test)
    ],
)


Path(
    "data/evaluation/models"
).mkdir(
    parents=True,
    exist_ok=True,
)


model.booster_.save_model(
    "data/evaluation/models/u100_residual_lgbm.txt"
)


pred = model.predict(X_test)


rmse = (
    ((pred - y_test) ** 2)
    .mean()
) ** 0.5


print(
    "RMSE:",
    rmse,
)
