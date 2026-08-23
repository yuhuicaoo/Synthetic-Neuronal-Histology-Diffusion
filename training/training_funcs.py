import torch
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm
from copy import deepcopy
from contextlib import contextmanager

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha = 1-self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        self.ema_model.load_state_dict(sd)

@contextmanager
def use_ema_weights(diffusion_model, ema_controlnet, ema_label_conditioner):
    controlnet_state = deepcopy(diffusion_model.controlnet.state_dict())
    label_conditioner_state = deepcopy(diffusion_model.label_conditioner.state_dict())

    diffusion_model.controlnet.load_state_dict(ema_controlnet.state_dict())
    diffusion_model.label_conditioner.load_state_dict(ema_label_conditioner.state_dict())
    diffusion_model.eval()

    try:
        yield diffusion_model
    finally:
        diffusion_model.controlnet.load_state_dict(controlnet_state)
        diffusion_model.label_conditioner.load_state_dict(label_conditioner_state)
        diffusion_model.train()

@torch.no_grad()
def sample_with_ema(
        diffusion_model,
        ema_controlnet,
        ema_label_conditioner,
        neuron_structure, 
        treatment_label,
        region_label,
        guidance_scale=2.0,
    ):

    with use_ema_weights(diffusion_model, ema_controlnet, ema_label_conditioner) as m:
        return m.sample(
            neuron_structure=neuron_structure,
            treatment_label=treatment_label,
            region_label=region_label,
            guidance_scale=guidance_scale
        )

@torch.no_grad()
def evaluate_kid(diffusion_model, ema_controlnet, ema_label_conditioner, loader, device, guidance_scale=2.0):

    kid = KernelInceptionDistance(subset_size=50, normalize=True).to(device)

    with use_ema_weights(diffusion_model, ema_controlnet, ema_label_conditioner) as m:
        for batch in tqdm(loader, desc="Computing KID score", leave=False):
            
            real = batch["image"].to(device)
            neuron_structure = batch["neuron_structure"].to(device)
            treatment_label = batch['treatment_label'].to(device)
            region_label = batch['region_label'].to(device)

            fake = m.sample(
                neuron_structure=neuron_structure,
                treatment_label=treatment_label,
                region_label=region_label,
                guidance_scale=guidance_scale,
            )

            # convert from [-1,1] -> [0,1] (generated images are already in range)
            real = (real + 1) / 2

            kid.update(real, real=True)
            kid.update(fake, real=False)

    kid_mean, kid_std = kid.compute()
    return kid_mean.item(), kid_std.item()