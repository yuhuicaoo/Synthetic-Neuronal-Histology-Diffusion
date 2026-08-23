from config import DiffusionConfig
from unet_model import ConditionedUnet_Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import cosine_beta_schedule

class Image_Diffusion(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.model = ConditionedUnet_Image(config=config).to(config.device)

        # self.beta = torch.linspace(config.beta_start, config.beta_end, config.time_steps).to(config.device)
        self.beta = cosine_beta_schedule(config.time_steps).to(config.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)   # (timesteps, )

    def noise_images(self, x, t):
        # reshape to match dimensions of x
        sqrt_alpha_hat = self.alpha_hat[t].sqrt()[:, None, None, None]                      # (batch, ) -> (batch, 1 ,1 ,1)
        sqrt_one_minus_alpha = (1 - self.alpha_hat[t]).sqrt()[:, None, None, None]          # (batch, ) -> (batch, 1 ,1 ,1)

        # generate random noise of shape x
        noise = torch.randn_like(x)             # (batch, in_channels, image_size, image_size)

        return (sqrt_alpha_hat * x) + (sqrt_one_minus_alpha * noise), noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.config.time_steps, size=(n,), device=self.config.device)

    def forward(self, x, nuclei_structure):
        timesteps = self.sample_timesteps(x.shape[0])

        # create noisy images and get the noise used
        x_t , noise = self.noise_images(x, timesteps)

        # predict the noise applied to the original image
        pred_noise = self.model(x_t, timesteps, nuclei_structure)

        return F.mse_loss(input = pred_noise, target = noise)

    @torch.no_grad()
    def sample(self, nuclei_structure):
        self.model.eval()

        # get batch_size
        B = nuclei_structure.shape[0]

        x = torch.randn(B, self.config.in_channels, self.config.image_size, self.config.image_size, device=self.config.device)

        for i in reversed(range(1, self.config.time_steps)):
            t = (torch.ones(B) * i).long().to(self.config.device)
            pred_noise = self.model(x, t, nuclei_structure)

            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

            x = (1 / alpha.sqrt()) * (x - ((1 - alpha) /  (1 - alpha_hat).sqrt()) * pred_noise) + beta.sqrt() * noise
        
        self.model.train()
        x = (x.clamp(-1, 1) + 1) / 2            # image [-1 , 1] -> [0, 1]
        return x
    
    @torch.no_grad()
    def ddim_sample(self, nuclei_structure, steps = 50, eta = 0.0):
        self.model.eval()

        B = nuclei_structure.shape[0]
        x = torch.randn(B, self.config.in_channels, self.config.image_size, self.config.image_size, device=self.config.device)

        step_ratio = self.config.time_steps // steps
        timesteps = list(reversed(range(0, self.config.time_steps, step_ratio)))

        for i, timestep in enumerate(timesteps):
            t = (torch.ones(B) * timestep).long().to(self.config.device)

            # predict noise
            pred_noise = self.model(x, t, nuclei_structure)

            alpha_hat_t = self.alpha_hat[t][:, None, None, None]

            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
            else:
                t_prev = -1
            
            if t_prev >= 0:
                alpha_hat_prev = self.alpha_hat[(torch.ones(B) * t_prev).long().to(self.config.device)][:, None, None, None]
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
        x = (x.clamp(-1, 1) + 1) / 2
        return x