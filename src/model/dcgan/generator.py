import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,
                 latent_channels,
                 hidden_channels=512,
                 n_image_channels=3,
                 image_size=64):
        
        super().__init__()

        # (latent_channels x 1 x 1)
        # -> (512 x 4 x 4)
        # -> (256 x 8 x 8)
        # -> (128 x 16 x 16)
        # -> (64 x 32 x 32)
        # -> (3 x 64 x 64)

        layers_list = [
            nn.Sequential(
                nn.ConvTranspose2d(
                    latent_channels, hidden_channels,
                    kernel_size = 4,
                    stride = 1,
                    padding = 0,
                    bias=False
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU()
            )
        ]
        current_size = 4

        while current_size != image_size // 2:
            layers_list.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_channels, hidden_channels // 2,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias=False
                    ),
                    nn.BatchNorm2d(hidden_channels // 2),
                    nn.ReLU()
                )
            )
            hidden_channels //= 2
            current_size *= 2

        layers_list.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_channels, n_image_channels,
                    kernel_size = 4,
                    stride = 2,
                    padding = 1,
                    bias = False
                ),
                nn.Tanh()
            )
        )

        self.layers = nn.Sequential(*layers_list)

    def forward(self, input):
        return self.layers(input)
