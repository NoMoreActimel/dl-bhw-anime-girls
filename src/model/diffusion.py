import nupmy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from .utils import _extract_into_tensor


class Diffusion:
    def __init__(self, *, betas: np.array, loss_type: str = "mse"):
        """
        Class that simulates Diffusion process. Does not store model or optimizer.
        """

        betas = torch.from_numpy(betas).double()
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.alphas = 1 - self.betas
        self.alphas_pred = torch.cat([torch.tensor([1.0]), self.alphas[:-1]], dim=0)
        self.sqrt_alphas = self.alphas.sqrt()
        self.sqrt_alphas_pred = self.alphas_pred.sqrt()

        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]), ], dim=0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.sqrt_alphas_cumprod[:-1]], dim=0)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_alphas_coeff = (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_variance =  posterior_alphas_coeff * self.betas

        # log calculation clipped because posterior variance is 0.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]], dim=0)
        )
        self.posterior_mean_coef1 = self.sqrt_alphas * posterior_alphas_coeff
        self.posterior_mean_coef2 = self.sqrt_alphas_cumprod_prev * self.betas / (1 - self.alphas_cumprod)

        if loss_type == "mse":
            self.loss = nn.MSELoss() # (reduction='none')
        else:
            raise NotImplementedError("Only MSE loss supported :(")

    def q_mean_variance(self, x0, t):
        """
        Get mean and variance of distribution q(x_t | x_0). Use equation (1).
        """
        # sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        # sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0
        variance = _extract_into_tensor(1 - self.alphas_cumprod, t, x0.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x0.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute mean and variance of diffusion posterior q(x_{t-1} | x_t, x_0).
        Use equation (2) and (3).
        """
        assert x_start.shape == x_t.shape

        posterior_mean_coef1 = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2 = _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = posterior_mean_coef1 * x_t + posterior_mean_coef2 * x_start

        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse data for a given number of diffusion steps.
        Sample from q(x_t | x_0).
        """
        noise = torch.randn_like(x_start) if noise is None else noise
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return mean + noise * std

    def p_mean_variance(self, model_output, x, t):
        """
        Apply model to get p(x_{t-1} | x_t). Use Equation (2) and plug in \hat{x}_0;
        """
        model_variance = torch.cat([self.posterior_variance[1:2], self.betas[1:]], dim=0)
        model_log_variance = torch.log(model_variance)
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        # \hat{x}_0 = (x_t - sqrt(1 - alphas_cumprod[t]) * eps) / sqrt(alphas_cumprod[t])
        pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
        posterior_mean_coef1 = _extract_into_tensor(self.posterior_mean_coef1, t, x.shape)
        posterior_mean_coef2 = _extract_into_tensor(self.posterior_mean_coef2, t, x.shape)
        model_mean = posterior_mean_coef1 * x + posterior_mean_coef2 * pred_xstart

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Get \hat{x0} from epsilon_{theta}. Use equation (4) to derive it.
        """
        sqrt_one_minus_alphas_cumprod = _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_alphas_cumprod = _extract_into_tensor(
            self.sqrt_alphas_cumprod, t, x_t.shape
        )
        x0_hat = (x_t - sqrt_one_minus_alphas_cumprod * eps) / sqrt_alphas_cumprod
        return x0_hat

    def p_sample(self, model_output, x, t):
        """
        Sample from p(x_{t-1} | x_t).
        """
        out = self.p_mean_variance(model_output, x, t) # get mean, variance of p(xt-1|xt)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0) # no noise when t == 0
        while nonzero_mask.dim() < noise.dim():
            nonzero_mask = nonzero_mask.unsqueeze(-1)

        # print(out["mean"].shape, nonzero_mask.shape, out["log_variance"].shape, noise.shape)

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample}

    def p_sample_loop(self, model, shape):
        """
        Samples a batch=shape[0] using diffusion model.
        """

        x = torch.randn(*shape, device=model.device)
        indices = list(range(self.num_timesteps))[::-1]

        for i in tqdm(indices):
            t = torch.tensor([i] * shape[0], device=x.device)
            with torch.no_grad():
                model_output = model(x, t)
                out = self.p_sample(model_output, x, t)
                x = out["sample"]
            
        return x

    def train_loss(self, model, x0):
        """
        Calculates loss L^{simple}_t for the given model, x0.
        """
        t = torch.randint(0, self.num_timesteps, size=(x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        model_output = model(xt, t)
        loss = self.loss(model_output, noise)
        # print(model_output.shape, x0.shape, loss)
        return loss, model_output
