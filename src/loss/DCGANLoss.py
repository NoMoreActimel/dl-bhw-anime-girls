import torch
import torch.nn as nn

class DCGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator_criterion = nn.BCELoss()
        self.generator_criterion = nn.BCELoss()
    
    def generator_loss(self, gen_predicts, **kwargs):
        batch_size = gen_predicts.shape[0]
        gen_targets = torch.ones(batch_size, 1, device=gen_predicts.device)
        loss = self.generator_criterion(gen_predicts, gen_targets)
        return loss

    def discriminator_loss(self, real_predicts, gen_predicts, **kwargs):
        batch_size = real_predicts.shape[0]

        real_targets = torch.ones(batch_size, 1, device=real_predicts.device)
        real_loss = self.discriminator_criterion(real_predicts, real_targets)

        gen_targets = torch.zeros(batch_size, 1, device=gen_predicts.device)
        gen_loss = self.discriminator_criterion(gen_predicts, gen_targets)

        return real_loss + gen_loss, real_loss, gen_loss
