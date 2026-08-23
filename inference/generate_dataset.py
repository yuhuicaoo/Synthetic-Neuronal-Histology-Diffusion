import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
os.environ['HF_HOME'] = '/eresearch/hie/ycao891/Yuhui/hf_cache'
os.environ['TORCH_HOME'] = '/eresearch/hie/ycao891/Yuhui/torch_cache'

import time
import math
import torch
import itertools
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from utils.util import watershed_postprocess
from model.mask_diffusion import Mask_Diffusion
from model.controlnetImage_diffusion import ControlNetImageDiffusion
from utils.config import DiffusionConfig, TREATMENT_GROUPS, TREATMENT_MAP, REGION_GROUPS, REGION_MAP

if __name__ == '__main__':
    save_dir = Path('Neuron Dataset Generated 4')
    os.makedirs(save_dir, exist_ok=True)

    model_base_path = Path('saves')
    image_model_path = model_base_path / f'image_training_3/image_epoch500.pth'
    mask_model_path = model_base_path / f'mask_training_2/model_best.pth'

    config = DiffusionConfig()

    # load checkpoints and models
    checkpoint_image = torch.load(image_model_path, map_location=config.device)
    checkpoint_mask = torch.load(mask_model_path, map_location=config.device)

    mask_model = Mask_Diffusion(config=config).to(config.device)
    mask_model.load_state_dict(checkpoint_mask['ema_model'])
    mask_model.eval()

    image_model = ControlNetImageDiffusion(
        config=config,
        pretrained_model='sd-legacy/stable-diffusion-v1-5',
        train_unet=False,
        unfreeze_last_n_down_blocks=0,
        use_unet_lora=False,
        lora_rank=4
    ).to(config.device)

    image_model.controlnet.load_state_dict(checkpoint_image['ema_controlnet'])
    image_model.label_conditioner.load_state_dict(checkpoint_image['ema_label_conditioner'])
    # image_model.unet.load_state_dict(checkpoint_image['unet_lora'], strict=False)
    image_model.eval()

    n_patches_per_combo = 20
    batch_size = 4
    label_combos = list(itertools.product(TREATMENT_GROUPS, REGION_GROUPS))

    for c in tqdm(label_combos, desc='Inference'):
        treatment, region = c
        output_save_dir = save_dir / treatment / region
        os.makedirs(output_save_dir, exist_ok=True)

        sample_idx = 1
        print(f"\nGenerating Samples at Treatment: {treatment} | Region: {region}")

        for i in range(math.ceil(n_patches_per_combo / batch_size)):
            current_batch_size = min(batch_size, n_patches_per_combo - i * batch_size)
            t = torch.full((current_batch_size,), TREATMENT_MAP[treatment], device=config.device, dtype=torch.long)
            r = torch.full((current_batch_size,), REGION_MAP[region], device=config.device, dtype=torch.long)

            # generate mask-image pair
            start = time.time()
            neuron_structure = mask_model.sample(
                n_samples=current_batch_size,
                treatment_label=t,
                region_label=r,
                steps=100,
                guidance_scale=3.0
            )

            image_samples = image_model.sample(
                neuron_structure, 
                treatment_label=t,
                region_label=r,
                steps=100, 
                guidance_scale=2.0
            )
            print(f"Generation time: {(time.time() - start):.2f} seconds")


            for b in range(current_batch_size):
                image = image_samples[b].cpu().permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)

                structure = neuron_structure[b].cpu().numpy()
                instance_mask = watershed_postprocess(structure, fg_threshold=0.5, hover_threshold=0.4, min_area=25)

                Image.fromarray(image).save(
                    output_save_dir / f'{sample_idx:04d}_img.tif'
                )
                Image.fromarray(instance_mask.astype(np.uint16)).save(
                    output_save_dir / f'{sample_idx:04d}_masks.tif'
                )

                sample_idx += 1

            del neuron_structure, image_samples
            torch.cuda.empty_cache()

