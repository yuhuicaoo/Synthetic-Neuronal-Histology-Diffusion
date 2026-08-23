import os

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
os.environ['HF_HOME'] = '/eresearch/hie/ycao891/Yuhui/hf_cache'
os.environ['TORCH_HOME'] = '/eresearch/hie/ycao891/Yuhui/torch_cache'

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm
from utils.util import get_patches_2
from pathlib import Path
from utils.config import TREATMENT_GROUPS, REGION_GROUPS
from utils.config import DiffusionConfig, PatchConfig
from model.controlnetImage_diffusion import ControlNetImageDiffusion
from data.create_dataset import NeuronPatchDataset
from data.load_data import load_image_mask_pairs

@torch.no_grad
def generate_from_real_structure(image_model, loader, device, guidance_scale):
    generated_patches = []
    
    for batch in tqdm(loader, desc='Generating from real neuron structure'):
        neuron_structure = batch["neuron_structure"].to(device)
        treatment_label = batch['treatment_label'].to(device)
        region_label = batch['region_label'].to(device)
        
        fake = image_model.sample(
            neuron_structure=neuron_structure,
            treatment_label=treatment_label,
            region_label=region_label,
            guidance_scale=guidance_scale,
        )

        fake_np = fake.permute(0, 2, 3, 1).cpu().numpy()

        for i in range(fake_np.shape[0]):
            generated_patches.append({
                "image": fake_np[i],
                "treatment": batch["treatment_label"][i],
                "region": batch["region_label"][i]
            })

    return generated_patches

def calc_fid_kid_scores(real_ds, gen_ds, batch_size, device):
    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(normalize=True, subset_size=50).to(device)

    def feed(dataset, metric, is_real):
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)
        for batch in tqdm(loader):
            imgs = torch.stack([
                torch.from_numpy(patch['image']).permute(2, 0, 1)
                for patch in batch
            ]).to(device)
            metric.update(imgs, real=is_real)

    feed(real_ds, fid, is_real=True)
    feed(gen_ds, fid, is_real=False)
    fid_score = fid.compute().item()

    feed(real_ds, kid, is_real=True)
    feed(gen_ds, kid, is_real=False)
    kid_mean, kid_std = kid.compute()

    return {
        "fid": fid_score,
        "kid_mean": kid_mean.item(),
        "kid_std": kid_std.item()
    }



if __name__ == "__main__":
    ds_gen_root = Path("Neuron Dataset Generated 3")
    ds_root = Path('Neuron Dataset')

    base_ds = load_image_mask_pairs(ds_root, TREATMENT_GROUPS, REGION_GROUPS)


    diffusion_config = DiffusionConfig()
    patch_config = PatchConfig(stride=256)
    image_model_path = Path('saves') / f'image_training_3/image_epoch500.pth'
    checkpoint_image = torch.load(image_model_path, map_location=diffusion_config.device)

    image_model = ControlNetImageDiffusion(
        config=diffusion_config,
        pretrained_model='sd-legacy/stable-diffusion-v1-5',
        train_unet=False,
        unfreeze_last_n_down_blocks=0,
        use_unet_lora=False,
        lora_rank=4
    ).to(diffusion_config.device)

    image_model.controlnet.load_state_dict(checkpoint_image['ema_controlnet'])
    image_model.label_conditioner.load_state_dict(checkpoint_image['ema_label_conditioner'])
    image_model.eval()

    real_patches = get_patches_2(base_ds, config=patch_config)
    real_patches_2 = NeuronPatchDataset(real_patches)

    real_dataloader = DataLoader(
        real_patches_2,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )

    gen_patches_from_real = generate_from_real_structure(
        image_model, 
        real_dataloader, 
        device=diffusion_config.device, 
        guidance_scale=2.0
    )

    scores_stage2_only = calc_fid_kid_scores(real_patches, gen_patches_from_real, batch_size=32, device=diffusion_config.device)
    print(scores_stage2_only)