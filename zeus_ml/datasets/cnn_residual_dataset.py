from pathlib import Path

import torch
from torch.utils.data import Dataset


class CNNResidualDataset(Dataset):
    """
    Lazy-loading dataset for Zeus multi-variable CNN residual training.

    Expected structure:

    cnn_residual/
        20250422T180000Z/
            input_000.pt
            target_000.pt
            input_001.pt
            target_001.pt
            ...

    Each sample:

        input:
            (4, 721, 1440)

            channels:
            0: 2m_temperature GFS
            1: 100m_u_component_of_wind GFS
            2: 100m_v_component_of_wind GFS
            3: surface_solar_radiation_downwards GFS


        target:
            (4, 721, 1440)

            channels:
            ERA5 - GFS residual
    """

    def __init__(
        self,
        root_dir: str,
        cycles=None,
    ):
        self.root_dir = Path(root_dir)

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}"
            )

        self.samples = []


        cycle_dirs = sorted(
            [
                p for p in self.root_dir.iterdir()
                if p.is_dir()
            ]
        )


        if cycles is not None:

            cycle_dirs = [
                p for p in cycle_dirs
                if p.name in cycles
            ]


        for cycle in cycle_dirs:

            inputs = sorted(
                cycle.glob(
                    "input_*.pt"
                )
            )

            for input_file in inputs:

                lead = input_file.stem.replace(
                    "input_",
                    ""
                )

                target_file = cycle / (
                    f"target_{lead}.pt"
                )


                if not target_file.exists():
                    raise FileNotFoundError(
                        f"Missing target: {target_file}"
                    )


                self.samples.append(
                    (
                        input_file,
                        target_file,
                    )
                )


        if len(self.samples) == 0:
            raise RuntimeError(
                "No CNN residual samples found."
            )


        print(
            "CNN samples:",
            len(self.samples)
        )


    def __len__(self):

        return len(self.samples)



    def __getitem__(
        self,
        index,
    ):

        input_file, target_file = (
            self.samples[index]
        )


        x = torch.load(
            input_file,
            weights_only=True,
        )

        y = torch.load(
            target_file,
            weights_only=True,
        )


        # float16 on disk
        # float32 for training

        x = x.float()

        y = y.float()


        return x, y