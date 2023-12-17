import math
import torch
import torch.nn as nn


class DCGANDiscriminator(nn.Module):
    def __init__(self, hidden_channels=512, n_image_channels=3, image_size=64):
        super().__init__()

        # last layer reduces size by the factor of 4, others by 2 
        # image_size = 64  =>  n_layers = 4, channels = hidden_channels // 8
        n_layers = round(math.log(image_size // 4))
        channels = hidden_channels // (2 ** (n_layers - 1))

        # 3 x 64 x 64
        # 64 x 32 x 32
        # 128 x 16 x 16
        # 256 x 8 x 8
        # 512 x 4 x 4
        # 1 x 1 x 1

        layers_list = [
            nn.Sequential(
                nn.Conv2d(
                    n_image_channels, channels,
                    kernel_size = 4,
                    stride = 2,
                    padding = 1,
                    bias = False
                ),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        ]

        while channels != hidden_channels:
            layers_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels, channels * 2,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False
                    ),
                    nn.BatchNorm2d(channels * 2),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            channels *= 2
        
        layers_list.append(
            nn.Sequential(
                nn.Conv2d(
                    hidden_channels, 1,
                    kernel_size = 4,
                    stride = 1,
                    padding = 0,
                    bias = False
                ),
                nn.Sigmoid()
            )
        )
        self.layers = nn.Sequential(*layers_list)

    def forward(self, input):
        return self.layers(input)