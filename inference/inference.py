import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['HF_HOME'] = '/eresearch/hie/ycao891/Yuhui/hf_cache'
os.environ['TORCH_HOME'] = '/eresearch/hie/ycao891/Yuhui/torch_cache'

import time
import torch
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from utils.util import watershed_postprocess
from model.mask_diffusion import Mask_Diffusion
from model.controlnetImage_diffusion import ControlNetImageDiffusion
from utils.config import DiffusionConfig, TREATMENT_GROUPS, TREATMENT_MAP, REGION_GROUPS, REGION_MAP


if __name__ == "__main__":
    save_dir = Path("saves")
    image_model_path = save_dir / f'image_training_3/image_epoch500.pth'
    mask_model_path = save_dir / f'mask_training_1/model_best.pth'

    inference_save_path = save_dir / f'inference_12'
    os.makedirs(inference_save_path, exist_ok=True)

    config = DiffusionConfig()

    checkpoint_image = torch.load(image_model_path, map_location=config.device)
    checkpoint_mask = torch.load(mask_model_path, map_location=config.device)

    image_model = ControlNetImageDiffusion(
        config = config,
        pretrained_model = 'sd-legacy/stable-diffusion-v1-5',
        train_unet = False,
        unfreeze_last_n_down_blocks = 0,
        use_unet_lora = False,
        lora_rank = 4
    ).to(config.device)

    image_model.controlnet.load_state_dict(checkpoint_image['ema_controlnet'])
    image_model.label_conditioner.load_state_dict(checkpoint_image['ema_label_conditioner'])
    # image_model.unet.load_state_dict(checkpoint_image['unet_lora'], strict=False)
    image_model.eval()

    mask_model = Mask_Diffusion(config=config).to(config.device)
    mask_model.load_state_dict(checkpoint_mask['ema_model'])
    mask_model.eval()

    n_samples = 4
    combinations = list(itertools.product(TREATMENT_GROUPS, REGION_GROUPS))

    col_titles = [
        'Binary Map', 
        'Horizontal Distance Map', 
        'Vertical Distance Map',
        'Instance Segmentation Mask', 
        'Image'
    ]

    for c in tqdm(combinations , desc='Inference'):
        treatment, region = c
        print(f"\nTreatment: {treatment} | Region: {region}")
        
        t_labels = torch.tensor(TREATMENT_MAP[treatment]).expand(n_samples).to(config.device)
        r_labels = torch.tensor(REGION_MAP[region]).expand(n_samples).to(config.device)

        # generate neuron structure
        print(f"Generating neuron structure samples")
        start = time.time()
        mask_samples = mask_model.sample(
            n_samples=n_samples,
            treatment_label=t_labels, 
            region_label=r_labels, 
            steps=50,
            guidance_scale=3.0
        )
        print(f"Generation time: {(time.time() - start):.2f} seconds")

        # generate image from neuron structure
        print(f"Generating image samples")
        start = time.time()
        image_samples = image_model.sample(
            mask_samples, 
            t_labels,
            r_labels,
            steps=50, 
            guidance_scale=2.0
        )
        print(f"Generation time: {(time.time() - start):.2f} seconds")
        
        mask_samples = mask_samples.cpu().numpy()
        image_samples = image_samples.cpu().numpy()

        instance_maps = []
        for i in range(mask_samples.shape[0]):
            instance_maps.append(watershed_postprocess(mask_samples[i]))

        fig , axes = plt.subplots(n_samples, 5, figsize=(16,16))

        for i in range(n_samples):

            # Column 1: Binary Mask
            axes[i, 0].imshow((mask_samples[i, 0] + 1) / 2)
            axes[i, 0].axis("off")

            # Column 2: Horizontal Distance Map
            axes[i, 1].imshow(mask_samples[i, 1])
            axes[i, 1].axis("off")

            # Column 3: Vertical Distance Map
            axes[i, 2].imshow(mask_samples[i, 2])
            axes[i, 2].axis("off")

            # Column 4: Instance Map
            axes[i, 3].imshow(instance_maps[i])
            axes[i, 3].axis('off')

            # Column 5: Generate Image:
            axes[i, 4].imshow(np.transpose(image_samples[i], (1, 2,0)))
            axes[i, 4].axis('off')

            if i == 0:
                for col, title in enumerate(col_titles):
                    axes[i, col].set_title(title)
            

        
        plt.tight_layout()
        plt.savefig(inference_save_path / f'{treatment}_{region}.png')
        print(f"Saved plots to {inference_save_path / f'{treatment}_{region}.png'}\n")
        plt.close(fig)

        del mask_samples, image_samples, t_labels, r_labels, instance_maps
        torch.cuda.empty_cache()



