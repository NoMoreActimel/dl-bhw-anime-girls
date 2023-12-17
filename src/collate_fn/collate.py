import logging
import torch

logger = logging.getLogger(__name__)

class CollateClass:
    def __call__(self, images):
        images = torch.tensor(images)
        return {"images": images}
