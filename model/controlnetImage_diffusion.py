import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DiffusionConfig
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler, DDIMScheduler
from peft import LoraConfig

class LabelConditioner(nn.Module):
    """
    Embeds (treatment_id, region_id) into a length-1 sequence matching the
    UNet's cross_attention_dim, standing in for CLIP text embeddings.
    """
 
    def __init__(self, num_treatments, num_regions, cross_attn_dim, embed_dim=256):
        super().__init__()
        self.num_treatments = num_treatments
        self.num_regions = num_regions
        # +1 to number of treatment / region class to represent null token for CFG
        self.treatment_emb = nn.Embedding(num_treatments + 1, embed_dim)
        self.region_emb = nn.Embedding(num_regions + 1, embed_dim)

        self.treatment_proj = nn.Sequential(
            nn.Linear(embed_dim, cross_attn_dim),
            nn.SiLU(),
            nn.Linear(cross_attn_dim, cross_attn_dim)
        )

        self.region_proj = nn.Sequential(
            nn.Linear(embed_dim, cross_attn_dim),
            nn.SiLU(),
            nn.Linear(cross_attn_dim, cross_attn_dim)
        )
 
    def forward(self, treatment_label=None, region_label=None, drop_mask=None, batch_size=None, device=None):
        if treatment_label is None or region_label is None:
            assert batch_size is not None and device is not None
            treatment_label = torch.full((batch_size,), self.num_treatments, device=device, dtype=torch.long)
            region_label = torch.full((batch_size,), self.num_regions, device=device, dtype=torch.long)
        elif drop_mask is not None:
            null_treatment = torch.full_like(treatment_label, self.num_treatments)
            null_region = torch.full_like(region_label, self.num_regions)
            treatment_label = torch.where(drop_mask, null_treatment, treatment_label)
            region_label = torch.where(drop_mask, null_region, region_label)

        t = self.treatment_proj(self.treatment_emb(treatment_label))        # (B, cross_atn_dim)
        r = self.region_proj(self.region_emb(region_label))                 # (B, cross_atn_dim)
        return torch.stack([t, r], dim = 1)                                 # (B, 2, cross_atn_dim)

class ControlNetImageDiffusion(nn.Module):
    def __init__(
            self, 
            config: DiffusionConfig, 
            pretrained_model: str = 'sd-legacy/stable-diffusion-v1-5',
            train_unet: bool = False,
            unfreeze_last_n_down_blocks: int = 0,
            use_unet_lora: bool = False,
            lora_rank: int = 8
        ):
        super().__init__()
        self.config = config
        self.device = config.device

        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet").to(self.device)

        self.controlnet = ControlNetModel.from_unet(self.unet, conditioning_channels=config.in_channels).to(self.device)

        self.train_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
        self.ddim_scheduler = DDIMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

        self.label_conditioner = LabelConditioner(
            num_treatments=config.num_treatments,
            num_regions=config.num_regions,
            cross_attn_dim=self.unet.config.cross_attention_dim
        ).to(self.device)

        self.vae.requires_grad_(False)
        self.vae.eval()

        self.unet.requires_grad_(False)
        if use_unet_lora:
            unet_lora_config = LoraConfig(
                r = lora_rank,
                lora_alpha = lora_rank,
                init_lora_weights = "gaussian",
                target_modules = ["to_k", "to_q", "to_v", "to_out.0"],           # attention projections
                lora_dropout=0.1
            )
            self.unet.add_adapter(unet_lora_config)
            self.unet.train()               # base weights remain frozen, only train LoRA parameters
        elif train_unet:
            self.unet.requires_grad_(True)
            self.unet.train()
        else:
            self.unet.eval()

        self._freeze_controlnet_backbone(unfreeze_last_n_down_blocks)

        self.controlnet.train()
        self.label_conditioner.train()

        self.scaling_factor = self.vae.config.scaling_factor

        self._print_trainable_params()

    def _freeze_controlnet_backbone(self, unfreeze_last_n_down_blocks=0):
        for p in self.controlnet.parameters():
            p.requires_grad_(False)
        
        for p in self.controlnet.controlnet_cond_embedding.parameters():
            p.requires_grad_(True)
        for p in self.controlnet.controlnet_down_blocks.parameters():
            p.requires_grad_(True)
        for p in self.controlnet.controlnet_mid_block.parameters():
            p.requires_grad_(True)
        for p in self.controlnet.conv_in.parameters():
            p.requires_grad_(True)

        if unfreeze_last_n_down_blocks > 0:
            for block in self.controlnet.down_blocks[-unfreeze_last_n_down_blocks:]:
                for p in block.parameters():
                    p.requires_grad_(True)

    def _print_trainable_params(self):
        total = sum(p.numel() for p in self.controlnet.parameters())
        trainable = sum(p.numel() for p in self.controlnet.parameters() if p.requires_grad)
        print(f"ControlNet: {trainable:,} / {total:,} trainable ({100*trainable/total:.1f}%)")

        unet_total = sum(p.numel() for p in self.unet.parameters())
        unet_trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        print(f"UNet: {unet_trainable:,} / {unet_total:,} trainable ({100*unet_trainable/unet_total:.1f}% )")

    def _encode(self, images):
        """
        images: expected to be [-1, 1] range
        """
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.scaling_factor

    def _decode(self, latents):
        with torch.no_grad():
            images = self.vae.decode(latents / self.scaling_factor).sample
        return images
    
    def _get_encoder_hidden_states(self, treatment_label, region_label, drop_mask=None):
        return self.label_conditioner(treatment_label, region_label, drop_mask=drop_mask)
    
    def forward(self, x, neuron_structure, treatment_label, region_label, cond_drop_prob = 0.1):
        latents = self._encode(x)
        bsz = latents.shape[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.train_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()
        noisy_latents = self.train_scheduler.add_noise(latents, noise, timesteps)

        # ranomly drop conditioning per-sample in the batch
        if cond_drop_prob > 0:
            drop_mask = (torch.rand(bsz, device=self.device) < cond_drop_prob)

            structure_drop_mask = drop_mask.view(bsz, 1, 1, 1)
            neuron_structure = torch.where(structure_drop_mask, torch.zeros_like(neuron_structure), neuron_structure)

            encoder_hidden_states = self._get_encoder_hidden_states(treatment_label, region_label, drop_mask=drop_mask)
        else:
            encoder_hidden_states = self._get_encoder_hidden_states(treatment_label, region_label)

        down_res, mid_res = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states = encoder_hidden_states,
            controlnet_cond = neuron_structure,
            return_dict = False,
        )

        pred_noise = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states = encoder_hidden_states,
            down_block_additional_residuals = down_res,
            mid_block_additional_residual = mid_res
        ).sample

        return F.mse_loss(pred_noise, noise)
    
    @torch.no_grad()
    def sample(self, neuron_structure, treatment_label = None, region_label = None, steps = 50, eta = 0.0, conditioning_scale = 1.0, guidance_scale = 1.0):
        self.controlnet.eval()
        B = neuron_structure.shape[0]

        latent_size = self.config.image_size // 8
        latents = torch.randn(B, self.unet.config.in_channels, latent_size, latent_size, device=self.device)

        if treatment_label is None or region_label is None:
            treatment_label = torch.full((B,), self.label_conditioner.num_treatments, device=self.device, dtype=torch.long)
            region_label = torch.full((B,), self.label_conditioner.num_regions, device=self.device, dtype=torch.long)

        encoder_hidden_states = self._get_encoder_hidden_states(treatment_label, region_label)

        use_cfg = guidance_scale != 1.0

        if use_cfg:
            null_cond = torch.zeros_like(neuron_structure)
            null_encoder_hidden_states = self.label_conditioner(batch_size=B, device=self.device)

            encoder_hidden_states_in = torch.cat([encoder_hidden_states, null_encoder_hidden_states], dim=0)
            cond_in = torch.cat([neuron_structure, null_cond], dim = 0)

        self.ddim_scheduler.set_timesteps(steps)
        for t in self.ddim_scheduler.timesteps:
            if use_cfg:
                latents_in = torch.cat([latents, latents], dim = 0)

                down_res, mid_res = self.controlnet(
                    latents_in,
                    t,
                    encoder_hidden_states = encoder_hidden_states_in,
                    controlnet_cond = cond_in,
                    conditioning_scale = conditioning_scale,
                    return_dict = False
                )
                noise_pred_both = self.unet(
                    latents_in,
                    t,
                    encoder_hidden_states = encoder_hidden_states_in,
                    down_block_additional_residuals = down_res,
                    mid_block_additional_residual = mid_res
                ).sample

                noise_pred_cond, noise_pred_uncond = noise_pred_both.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                down_res_u , mid_res_u = self.controlnet(
                    latents,
                    t,
                    encoder_hidden_states = encoder_hidden_states,
                    controlnet_cond = neuron_structure,
                    conditioning_scale = conditioning_scale,
                    return_dict = False
                )
                noise_pred = self.unet(
                    latents, 
                    t,
                    encoder_hidden_states = encoder_hidden_states,
                    down_block_additional_residuals = down_res_u,
                    mid_block_additional_residual = mid_res_u
                ).sample

            latents = self.ddim_scheduler.step(noise_pred, t, latents, eta=eta).prev_sample
        
        self.controlnet.train()
        images = self._decode(latents)
        return (images.clamp(-1, 1) + 1) / 2            # [0, 1]