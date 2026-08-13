from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from zeus_ml.datasets.direct_cnn_residual_dataset import (
    DirectCNNResidualDataset,
)

from zeus_ml.models.residual_cnn import ZeusResidualCNN


# -------------------------
# Configuration
# -------------------------

BUNDLE_ROOT = Path(
    "/Zeus/data/evaluation/forecast_store_hist/bundles"
)

ERA5_PATH = (
    "/Zeus/data/evaluation/era5"
)

OUT = (
    "/Zeus/data/evaluation/training/"
    "zeus_residual_cnn_30cycles.pt"
)


# First 30 cycles after holdout
TRAIN_START = 7
TRAIN_END = 19


EPOCHS = 1

BATCH_SIZE = 1

LEARNING_RATE = 1e-4


# -------------------------
# CPU settings
# -------------------------

torch.set_num_threads(8)


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Device:",
    device
)


# -------------------------
# Bundles
# -------------------------

all_bundles = sorted(
    BUNDLE_ROOT.glob("*")
)


train_bundles = all_bundles[
    TRAIN_START:
    TRAIN_END
]


print(
    "Training cycles:",
    len(train_bundles)
)


for b in train_bundles[:3]:
    print(
        "Example:",
        b.name
    )


# -------------------------
# Dataset
# -------------------------

dataset = DirectCNNResidualDataset(
    bundles=train_bundles,
    era5_path=ERA5_PATH,
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)


print(
    "Samples:",
    len(dataset)
)


# -------------------------
# Model
# -------------------------

model = ZeusResidualCNN(
    in_channels=4,
    out_channels=4,
    hidden_channels=16,
)


model = model.to(
    device
)


# -------------------------
# Weighted residual loss
# -------------------------

class MultiVariableResidualLoss(nn.Module):

    def __init__(self):

        super().__init__()

        # Channels:
        #
        # 0 temperature
        # 1 u100
        # 2 v100
        # 3 ssrd

        self.register_buffer(
            "weights",
            torch.tensor(
                [
                    1.0,
                    5.0,
                    5.0,
                    0.01,
                ],
                dtype=torch.float32,
            )
        )


    def forward(
        self,
        prediction,
        target,
    ):

        error = (
            prediction -
            target
        ) ** 2


        # average latitude/longitude

        loss = error.mean(
            dim=(2, 3)
        )


        loss = (
            loss *
            self.weights
        )


        return loss.mean()



criterion = MultiVariableResidualLoss()

criterion = criterion.to(
    device
)


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5,
)


# -------------------------
# Training
# -------------------------

for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0


    for step, (x, y) in enumerate(loader):

        x = x.to(
            device
        )

        y = y.to(
            device
        )


        optimizer.zero_grad()


        prediction = model(
            x
        )


        loss = criterion(
            prediction,
            y,
        )


        loss.backward()


        optimizer.step()


        total_loss += (
            loss.item()
        )


        if step % 10 == 0:

            print(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"Step {step}/{len(loader)} "
                f"Loss {loss.item():.6f}"
            )


    average_loss = (
        total_loss /
        len(loader)
    )


    print(
        f"Epoch {epoch+1} "
        f"Average loss: {average_loss:.6f}"
    )


# -------------------------
# Save
# -------------------------

Path(
    OUT
).parent.mkdir(
    parents=True,
    exist_ok=True,
)


torch.save(
    model.state_dict(),
    OUT,
)


print(
    "Saved:",
    OUT
)