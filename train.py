import argparse
import collections
import warnings

import numpy as np
import torch

import src.metric as module_metric

from src.model import UNet, Diffusion, get_named_beta_schedule, get_number_of_module_parameters
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    T = config["diffusion"]["n_timesteps"]

    # build model architecture, then print to console
    model = UNet(n_steps=T, time_emb_dim=config["model"]["time_emb_dim"])
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    print(f"\nNumber of model parameters: {get_number_of_module_parameters(model)}\n")

    schedule_type = config["diffusion"].get("schedule_type", "linear")
    betas = get_named_beta_schedule(schedule_type, T)
    diffusion = Diffusion(betas=betas, loss_type="mse")

    criterion = None

    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["metrics"]
    ]

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, params)
    lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    if "val" in config["data"]:
        inference_on_evaluation = config["data"]["val"].get("inference_on_evaluation", None)
        if inference_on_evaluation:
            inference_indices = config["data"]["val"]["inference_indices"]
    else:
        inference_on_evaluation = None
        inference_indices = None

    trainer = Trainer(
        model,
        diffusion,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None),
        len_val_epoch=config["trainer"].get("len_val_epoch", None),
        inference_on_evaluation=inference_on_evaluation,
        inference_indices=inference_indices
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"
        ),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
