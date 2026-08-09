from pathlib import Path
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(
    "/Zeus/data/evaluation/residual_ml/cycles"
)

OUT = Path(
    "/Zeus/data/evaluation/training/u100_train_all.parquet"
)

SAMPLES_PER_CYCLE = 5000000


def process_cycle(prefix):

    print("Loading:", prefix)

    gfs = torch.load(
        ROOT / f"{prefix}_gfs.pt"
    ).float()

    residual = torch.load(
        ROOT / f"{prefix}_residual.pt"
    )

    n = SAMPLES_PER_CYCLE

    t = torch.randint(0, gfs.shape[0], (n,))
    y = torch.randint(0, gfs.shape[1], (n,))
    x = torch.randint(0, gfs.shape[2], (n,))

    return pd.DataFrame(
        {
            "gfs_value": gfs[t,y,x].numpy(),
            "residual": residual[t,y,x].numpy(),
            "lead_hour": t.numpy(),
            "latitude": y.numpy(),
            "longitude": x.numpy(),
        }
    )


OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

writer = None

for gfs_file in sorted(ROOT.glob("*_gfs.pt")):

    prefix = gfs_file.name.replace(
        "_gfs.pt",
        ""
    )

    df = process_cycle(prefix)

    table = pa.Table.from_pandas(df)

    if writer is None:
        writer = pq.ParquetWriter(
            OUT,
            table.schema,
            compression="snappy"
        )

    writer.write_table(table)

    print(
        "written rows:",
        len(df)
    )


if writer:
    writer.close()


print("saved:", OUT)
