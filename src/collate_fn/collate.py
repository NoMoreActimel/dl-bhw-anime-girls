import logging
import torch

logger = logging.getLogger(__name__)

class CollateClass:
    def __call__(self, images):
        images = torch.cat([image.unsqueeze(0) for image in images])
        return {"images": images}
