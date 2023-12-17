from src.model.ddpm.model import UNet
from src.model.ddpm.diffusion import Diffusion
from src.model.ddpm.utils import get_named_beta_schedule, get_number_of_module_parameters

from src.model.dcgan.model import DCGAN
from src.model.dcgan.generator import DCGANGenerator
from src.model.dcgan.discriminator import DCGANDiscriminator

__all__ = [
    "UNet",
    "Diffusion",
    "get_named_beta_schedule"
    "get_number_of_module_parameters",
    "DCGAN",
    "DCGANGenerator",
    "DCGANDiscriminator"
]
