import argparse
import torch
import piq

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, Dataset

import src.model as module_model
import src.loss as module_loss

from src.model import Diffusion, get_named_beta_schedule, get_number_of_module_parameters
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser


def move_batch_to_device(batch, device: torch.device):
    """
    Move all necessary tensors to the HPU
    """
    for tensor_for_gpu in ["images"]:
        batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
    return batch


class GenDataset(Dataset):
    def __init__(self, generated):
        super().__init__()
        self.generated = generated
    
    def __getitem__(self, ind):
        return self.generated[ind]
    
    def __len__(self):
        return len(self.generated)


def run_inference(
        model_type,
        model,
        criterion,
        dataloader,
        diffusion=None):
    
    device = next(model.parameters()).device
    
    losses = []
    ssims = []
    generated = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing on dataloader", total=len(dataloader)):
            batch = move_batch_to_device(batch, device)
            if model_type == "DCGAN":    
                batch["generated"] = model(batch_size=batch["images"].shape[0])
                batch["gen_predicts"] = model.discriminate(batch["generated"])
                batch["loss"] = criterion.generator_loss(**batch)
            else:
                batch["loss"], batch["generated"] = diffusion.train_loss(model, batch["images"])
                
            batch["generated"] = 0.5 + 0.5 * batch["generated"]
            
            losses.append(batch["loss"])
            ssims.append(piq.ssim(batch["images"], batch["generated"], data_range=1.))
            generated.append(batch["generated"])


    if torch.cuda.is_available():
        gen_dataset = GenDataset(generated)
        gen_dataloader = DataLoader(gen_dataset, batch_size=1, collate_fn=lambda x: {"images": torch.cat(x)})

        fid_metric = piq.FID()
        real_feats = fid_metric.compute_feats(dataloader)
        gen_feats = fid_metric.compute_feats(gen_dataloader)
        fid = fid_metric(real_feats, gen_feats).item()
        print(f"FID: {fid:.4f}")
    else:
        print(f"No cuda is available, could not compute FID score")

    loss = sum(losses) / len(losses)
    print(f"Mean loss: {loss:.4f}")
    
    ssim = sum(ssims) / len(ssims)
    print(f"Mean SSIM: {ssim:.4f}")
        
        

def main(config):
    dataloaders = get_dataloaders(config)

    model_type = config["model_type"]
    assert model_type in ["DCGAN", "DDPM"], \
        f"Only DCGAN or DDPM model_type supported!"

    model = config.init_obj(config["model"], module_model)   

    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        
    checkpoint = torch.load(config.resume, device)
    if checkpoint["config"]["model"] != config["model"]:
        print(
            "Warning: Architecture configuration given in config file is different from that "
            "of checkpoint. This may yield an exception while state_dict is being loaded."
        )
    model.load_state_dict(checkpoint["state_dict"])
    print("Loaded model from checkpoint!")
    
    print(f"\nNumber of model parameters: {get_number_of_module_parameters(model)}\n")

    if model_type == "DDPM":
        T = config["diffusion"]["n_timesteps"]
        schedule_type = config["diffusion"].get("schedule_type", "linear")
        betas = get_named_beta_schedule(schedule_type, T)
        diffusion = Diffusion(betas=betas, loss_type="mse")
        criterion = None
    else:
        diffusion=None
        criterion = config.init_obj(config["loss"], module_loss)
    
    run_inference(
        model_type,
        model,
        criterion,
        dataloaders["val"],
        diffusion
    )


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
    config = ConfigParser.from_args(args)
    main(config)
