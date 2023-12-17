import torch
import torch.nn as nn

from src.model.dcgan.generator import DCGANGenerator
from src.model.dcgan.discriminator import DCGANDiscriminator


class DCGAN(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.model_config = model_config
        self.latent_channels = model_config["latent_channels"]
        self.hidden_channels = model_config["hidden_channels"]
        self.n_image_channels = model_config.get("n_image_channels", 3)
        self.image_size = model_config["image_size"]

        self.generator = DCGANGenerator(
            latent_channels=self.latent_channels,
            hidden_channels=self.hidden_channels,
            n_image_channels=self.n_image_channels,
            image_size=self.image_size
        )

        self.discriminator = DCGANDiscriminator(
            hidden_channels=self.hidden_channels,
            n_image_channels=self.n_image_channels,
            image_size=self.image_size
        )

    def forward(self, batch_size):
        noise = torch.randn(batch_size, self.latent_channels, 1, 1, device=self.device)
        return self.generator(noise)

    def discriminate(self, input):
        return self.discriminator(input)
