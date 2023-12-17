import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import librosa
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(
            self,
            model_type,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            diffusion=None,
            lr_scheduler=None,
            len_epoch=None,
            len_val_epoch=None,
            skip_oom=True,
            inference_on_evaluation=False,
            inference_indices=None
    ):
        """
            model_type: str, supported either DCGAN or DDPM
        """

        super().__init__(
            model_type, model, criterion, metrics,
            optimizer, lr_scheduler, config, device
        )

        self.diffusion = diffusion

        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders.get("val", None)

        if len_epoch is None:
            self.len_epoch = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.len_val_epoch = len_val_epoch
        if self.len_val_epoch:
            self.evaluation_dataloaders = {k: inf_loop(v) for k, v in self.evaluation_dataloaders}
        self.log_step = 50

        if self.model_type == "DCGAN":
            obl_train_metrics = [
                "generator_loss", "discriminator_loss", 
                "discriminator_real_loss", "discriminator_fake_loss",
                "generator_grad_norm", "discriminator_grad_norm"
            ]
            obl_val_metrics = [
                "generator_loss", "discriminator_loss", 
                "discriminator_real_loss", "discriminator_fake_loss"
            ]
        else:
            obl_train_metrics = ["loss", "grad_norm"]
            obl_val_metrics = ["loss"]

        self.train_metrics = MetricTracker(
            *(obl_train_metrics + [m.name for m in self.metrics if self._compute_on_train(m)]),
            writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            *(obl_val_metrics + [m.name for m in self.metrics]),
            writer=self.writer
        )

        self.inference_on_evaluation = inference_on_evaluation
        self.inference_indices = inference_indices
        self.val_dataset = self.val_dataloader.dataset if self.val_dataloader else None
        self.latent_noise = torch.randn(*self.val_dataset[0].shape, device=self.model.device)

    @staticmethod
    def _compute_on_train(metric):
        if hasattr(metric, "compute_on_train"):
            return metric.compute_on_train
        return True

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["images"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self, model):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc="train", total=self.len_epoch)):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics_tracker=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    raise e
                else:
                    raise e
            
            if self.model_type != "DCGAN":
                self.train_metrics.update("grad norm", self.get_grad_norm(self.model))
            
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                if self.model_type == "DCGAN":
                    self.logger.debug(
                        "Train Epoch: {} {} Generator Loss: {:.6f}, Discriminator Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx),
                            batch["generator_loss"].item(),
                            batch["discriminator_loss"].item()
                        )
                    )
                else:
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx),
                            batch["loss"].item()
                        )
                    )
                
                self._log_lr()
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx >= self.len_epoch:
                break
    
        log = last_train_metrics
        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    
    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()

        len_val_epoch = len(dataloader) if self.len_val_epoch is None else self.len_val_epoch

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc=part, total=len_val_epoch):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics_tracker=self.evaluation_metrics
                )

                if batch_idx >= len_val_epoch:
                    break
            
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
        
            if self.inference_on_evaluation:
                self._log_inference()

        if self.lr_scheduler is not None:
            if self.model_type == "DCGAN":
                for module_name in ["generator", "discriminator"]:
                    if not isinstance(self.lr_scheduler[module_name], ReduceLROnPlateau):
                        self.lr_scheduler[module_name].step()
            else:
                if not isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step()

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p.float(), bins="auto")
        
        return self.evaluation_metrics.result()


    def process_batch(self, batch, is_train: bool, metrics_tracker: MetricTracker):
        if self.model_type == "DCGAN":
            self.process_batch_dcgan(batch, is_train, metrics_tracker)
        elif self.model_type == "DDPM":
            self.process_batch_ddpm(batch, is_train, metrics_tracker)
        else:
            raise NotImplementedError("Only DCGAN and DDPM models are supported!")
    

    def process_batch_ddpm(self, batch, is_train: bool, metrics_tracker: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        
        batch["loss"], batch["generated"] = self.diffusion.train_loss(self.model, batch["images"])

        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm(self.model)
            self.optimizer.step()
        
        metrics_tracker.update("loss", batch["loss"].item())
        for met in self.metrics:
            metrics_tracker.update(met.name, met(**batch))
        
        return batch


    def process_batch_dcgan(self, batch, is_train: bool, metrics_tracker: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        batch["generated"] = self.model(batch["images"].shape[0])

        self.optimizer["discriminator"].zero_grad()
        batch["real_predicts"] = self.model.discriminate(batch["images"])
        batch["gen_predicts"] = self.model.disctiminate(batch["generated"].detach())

        if not is_train:        
            for met in self.metrics:
                metrics_tracker.update(met.name, met(**batch))
            return batch

        discr_losses = self.criterion.discriminator_loss(**batch)
        discriminator_loss_names = ["discriminator_loss", "real_loss", "gen_loss"]
        for loss_name, loss in zip(discriminator_loss_names, discr_losses):
            batch[loss_name] = loss
        batch["discriminator_loss"].backward()

        self._clip_grad_norm(self.model.discriminator)
        discriminator_grad_norm = self.get_grad_norm(self.model.discriminator)
        self.optimizer["discriminator"].step()
    
        self.optimizer["generator"].zero_grad()
        batch["gen_predicts"] = self.model.disctiminate(batch["generated"])
        batch["generator_loss"] = self.criterion.generator_loss(**batch)
        batch["generator_loss"].backward()

        self._clip_grad_norm(self.model.generator)
        generator_grad_norm = self.get_grad_norm(self.generator)
        self.optimizer["generator"].step()

        metrics_tracker.update("generator_loss", batch["generator_loss"].item())
        for loss_name in discriminator_loss_names:
            metrics_tracker.update(loss_name, batch[loss_name].item())
        
        metrics_tracker.update("generator_grad_norm", generator_grad_norm)
        metrics_tracker.update("discriminator_grad_norm", discriminator_grad_norm)

        for met in self.metrics:
            metrics_tracker.update(met.name, met(**batch))

        return batch


    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_text(self, text, name="text"):
        self.writer.add_text(name, text)
    
    def _log_image(self, image, name="image"):
        self.writer.add_image(name, image)

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _log_lr(self):
        if self.model_type == "DCGAN":
            last_lr = {}
            for model_name in ["generator", "discriminator"]:
                if not isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    last_lr[model_name] = self.optimizer[model_name].param_groups[0]['lr']
                else:
                    last_lr[model_name] = self.lr_scheduler[model_name].get_last_lr()[0]

            self.writer.add_scalar("generator learning rate", last_lr["generator"])
            self.writer.add_scalar("discriminator learning rate", last_lr["discriminator"])
            return

        if not isinstance(self.lr_scheduler, ReduceLROnPlateau):
            last_lr = self.optimizer.param_groups[0]['lr']
        else:
            last_lr = self.lr_scheduler.get_last_lr()[0]

        self.writer.add_scalar("learning rate", last_lr)

    def _log_inference(self):
        if self.model_type == "DCGAN":
            generated = self.model(self.latent_noise)
            self._log_image(generated, name="generation example")
            return
        
        for ind in self.infrence_indices:
            image = self.val_dataset[ind]
            reconstructed = self.diffusion.train_loss(self.model, image)
            self._log_image(image, name=f"original_{ind}")
            self._log_image(reconstructed, name=f"reconstructed_{ind}")
        
        generated = self.diffusion.p_sample_loop(
            self.model,
            shape=self.val_dataset[0].shape,
            noise=self.latent_noise
        )
        self._log_image(generated, name="generation example")