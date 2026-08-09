from datetime import datetime, timezone
from pathlib import Path

import torch

from forecast.forecast_store import ForecastStore
from zeus.utils.compression import decompress_prediction


cycle = datetime(
    2025,
    4,
    22,
    18,
    tzinfo=timezone.utc,
)

state_key = "100m_u_component_of_wind@0_360"

store = ForecastStore(
    Path(
        "/Zeus/data/evaluation/forecast_store_hist"
    )
)

payload = store.load_artifact(
    cycle,
    state_key,
)

tensor = decompress_prediction(
    payload,
    torch.Size(
        [
            361,
            721,
            1440,
        ]
    ),
)

print(
    tensor.shape,
    tensor.dtype,
)


out = Path(
    "/Zeus/data/evaluation/residual_ml/u100_gfs.pt"
)

torch.save(
    tensor.float(),
    out,
)

print(
    "saved:",
    out,
)
