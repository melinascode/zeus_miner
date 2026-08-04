from pathlib import Path
from datetime import datetime

from evaluation.truth import Era5TruthLoader


def load_era5(
    files,
    variable,
    cycle_time,
    horizon=360,
):

    loader = Era5TruthLoader()

    result = loader.load(
        files,
        variable=variable,
        cycle_time=cycle_time,
        horizon_hours=horizon,
    )

    return result.tensor