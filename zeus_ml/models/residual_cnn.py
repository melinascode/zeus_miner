import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(channels),
        )

        self.relu = nn.ReLU()


    def forward(self, x):

        return self.relu(
            x + self.block(x)
        )



class ZeusResidualCNN(nn.Module):

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        hidden_channels=64,
    ):
        super().__init__()


        self.encoder = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.ReLU(),
        )


        self.residual_layers = nn.Sequential(

            ResidualBlock(
                hidden_channels
            ),

            ResidualBlock(
                hidden_channels
            ),

            ResidualBlock(
                hidden_channels
            ),
        )


        self.decoder = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )



    def forward(self, x):

        x = self.encoder(x)

        x = self.residual_layers(x)

        residual = self.decoder(x)

        return residual