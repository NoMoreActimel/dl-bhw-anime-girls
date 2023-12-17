import torch
import torch.nn as nn

from src.model.dcgan.generator import DCGANGenerator
from src.model.dcgan.discriminator import DCGANDiscriminator


class DCGAN(nn.Module):
    def __init__(self, latent_channels, hidden_channels, image_size, n_image_channels=3, **kwargs):
        super().__init__()
        self.latent_channels = latent_channels
        self.generator = DCGANGenerator(
            latent_channels=self.latent_channels,
            hidden_channels=hidden_channels,
            n_image_channels=n_image_channels,
            image_size=image_size
        )
        self.discriminator = DCGANDiscriminator(
            hidden_channels=hidden_channels,
            n_image_channels=n_image_channels,
            image_size=image_size
        )

    def forward(self, batch_size):
        noise = torch.randn(batch_size, self.latent_channels, 1, 1, device=self.device)
        return self.generator(noise)

    def discriminate(self, input):
        return self.discriminator(input)
