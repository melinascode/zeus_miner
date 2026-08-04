from pathlib import Path
from datetime import datetime, timedelta


def find_era5_files(
    era5_root: str,
    variable: str,
    cycle_time: datetime,
    horizon_hours: int,
):
    """
    Find ERA5 daily files covering Zeus forecast verification window.
    """

    folder = Path(era5_root) / variable

    start_date = cycle_time.date()
    end_date = (
        cycle_time + timedelta(hours=horizon_hours)
    ).date()

    files = []

    current = start_date

    while current <= end_date:
        file_path = (
            folder /
            f"era5_{current.isoformat()}.nc"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing ERA5 file: {file_path}"
            )

        files.append(str(file_path))

        current += timedelta(days=1)

    return files