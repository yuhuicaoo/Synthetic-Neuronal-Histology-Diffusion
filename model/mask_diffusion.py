import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DiffusionConfig
from model.unet_model import ConditionedUnet, ConditionedUnet_Image
from util import cosine_beta_schedule

class Mask_Diffusion(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.model = ConditionedUnet(config).to(config.device)

        
        # self.beta = torch.linspace(config.beta_start, config.beta_end, config.time_steps).to(config.device)
        self.beta = cosine_beta_schedule(config.time_steps).to(config.device)
        self.alpha = 1 - self.beta
        # cumalative product of self.alpha up to alpha[t]
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)           # (timesteps, )

        self._print_trainable_params()

    def _print_trainable_params(self):
            total = sum(p.numel() for p in self.model.parameters())
            print(f"U-Net parameters: {total:,}")

    def _noise_image(self, x, t):
        """"""
        # reshape to match dimensions of x
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]                     # (batch, ) -> (batch, 1 ,1 ,1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]       # (batch, ) -> (batch, 1, 1 ,1)

        # generate random noise of shape x
        noise = torch.randn_like(x)          # (batch, in_channels, image_size, image_size)

        return (sqrt_alpha_hat * x) + (sqrt_one_minus_alpha_hat * noise), noise
    
    def _sample_timesteps(self, n):
        return torch.randint(low=1, high=self.config.time_steps, size=(n,), device=self.config.device)
    
    def forward(self, x, treatment_label, region_label, cond_drop_prob=0.1):
        timesteps = self._sample_timesteps(x.shape[0])

        # create noisy images and get the noise used
        x_t , noise = self._noise_image(x, timesteps)

        # classifier free guidance
        if cond_drop_prob > 0:
            drop_mask = torch.rand(x.shape[0], device=self.config.device) < cond_drop_prob
            treatment_label = torch.where(drop_mask, torch.full_like(treatment_label, self.model.num_treatments), treatment_label)
            region_label = torch.where(drop_mask, torch.full_like(region_label, self.model.num_regions), region_label)

        # predict the noise applied to the original image
        pred_noise = self.model(x_t, timesteps, treatment_label, region_label)
        return F.mse_loss(input = pred_noise, target = noise)
      
    @torch.no_grad()
    def sample(self, n_samples, treatment_label, region_label, steps = 50, eta = 0.0, guidance_scale=1.0):
        """Sampling using DDIM method"""
        self.model.eval()

        # start from random noise
        x = torch.randn(n_samples, self.config.in_channels, self.config.image_size, self.config.image_size, device=self.config.device)      # (n_samples, 3, 256, 256)

        null_treatment = torch.full_like(treatment_label, self.model.num_treatments)
        null_region = torch.full_like(region_label, self.model.num_regions)

        use_cfg = guidance_scale != 1.0
        if use_cfg:
            treatment_in = torch.cat([treatment_label, null_treatment], dim=0)
            region_in = torch.cat([region_label, null_region], dim=0)

        step_ratio = self.config.time_steps // steps
        timesteps = list(reversed(range(0, self.config.time_steps, step_ratio)))

        for i, timestep in enumerate(timesteps):
            # create timestep tensor
            t = (torch.ones(n_samples) * timestep).long().to(self.config.device)

            if use_cfg:
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t, t], dim=0)

                pred_noise_both = self.model(x_in, t_in, treatment_in, region_in)
                pred_noise_cond, pred_noise_uncond = pred_noise_both.chunk(2, dim=0)

                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            else:
                pred_noise_cond = self.model(x, t, treatment_label, region_label)

            alpha_hat_t = self.alpha_hat[t][:, None, None, None]

            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
            else:
                t_prev = -1
            
            if t_prev >= 0:
                alpha_hat_prev = self.alpha_hat[(torch.ones(n_samples) * t_prev).long().to(self.config.device)][:, None, None, None]
            else:
                alpha_hat_prev = torch.ones_like(alpha_hat_t)

            # update - predict x0 from current noisy image
            x0_pred = (x - torch.sqrt(1-alpha_hat_t) * pred_noise) / torch.sqrt(alpha_hat_t)
            x0_pred = x0_pred.clamp(-1, 1)

            # calc sigma
            sigma = eta * torch.sqrt((1 - alpha_hat_prev) / (1 - alpha_hat_t) * (1 - alpha_hat_t / alpha_hat_prev))

            # direction pointing to x_t
            direction = torch.sqrt(1 - alpha_hat_prev - sigma**2) * pred_noise

            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            
            x = torch.sqrt(alpha_hat_prev) * x0_pred + direction + sigma * noise
        
        self.model.train()
        output = x.clone().float()
        output[:, 0] = x[:, 0].clamp(-1, 1)                 # Binary map [-1, 1]
        output[:, 1] = x[:, 1].clamp(-1, 1)                 # Horiztonal distance map [-1, 1]
        output[:, 2] = x[:, 2].clamp(-1, 1)                 # Vertical distance map [-1, 1]
        return output
