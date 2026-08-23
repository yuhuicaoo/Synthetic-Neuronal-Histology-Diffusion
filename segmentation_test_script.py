import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
os.environ['HF_HOME'] = '/eresearch/hie/ycao891/Yuhui/hf_cache'
os.environ['TORCH_HOME'] = '/eresearch/hie/ycao891/Yuhui/torch_cache'

from load_data import load_image_mask_pairs, get_split_for_fold
from pathlib import Path
from util import collate_fn_segm, get_patches
import torch
from config import TREATMENT_GROUPS, REGION_GROUPS, PatchConfig
from torch.utils.data import DataLoader
from create_dataset import PatchDataset
from model.swin_maskRCNN import create_maskrcnn_resnet50
from train_segmentation import evaluate_segmentation_model
import numpy as np
import random

if __name__ == "__main__":
    ds_root = Path("Neuron Dataset")
    save_dir = Path('saves/segmentation_training_75_gen_2')
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = PatchConfig()
    num_folds = 5

    base_ds = load_image_mask_pairs(ds_root, TREATMENT_GROUPS, REGION_GROUPS)
    _, test_ds = get_split_for_fold(base_ds, folds=num_folds, test_size=0.2, seed=42)

    test_patches = get_patches(test_ds, config=config)
    print(f"Test patches: {len(test_patches)}")


    test_loader = DataLoader(
        PatchDataset(test_patches, train=False, use_albu=False),
        batch_size = 32,
        shuffle = False,
        num_workers = 4,
        pin_memory = True,
        persistent_workers = True,
        collate_fn = collate_fn_segm
    )

    fold_results = []
    for fold_idx in range(num_folds):
        print(f"\n === Fold {fold_idx} ===")
        
        model = create_maskrcnn_resnet50(num_classes=2, min_size=256, max_size=256)

        checkpoint = torch.load(f'{save_dir}/resnet_maskRCNN_fold{fold_idx}.pth', map_location=device, weights_only=True)

        model.load_state_dict(checkpoint)
        results = evaluate_segmentation_model(model, test_loader, device=device)
        fold_results.append(results)

        del model, checkpoint
        torch.cuda.empty_cache()

    metric_names = fold_results[0].keys()
    with open(f"{save_dir}/testing_results.txt", "w") as f:
        f.write(f"Resnet Backbone on MaskRCNN testing\n")
        f.write("Testing Results:\n\n\n")

        header = f"{'Metric':<10}"
        for fold in range(num_folds):
            header += f"{f'Fold {fold+1}':>10}"
        header += f"{'Mean':>10}{'Std':>10}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")

        for metric in metric_names:
            values = [r[metric] for r in fold_results]
            line = f"{metric:<10}"
            for value in values:
                line += f"{value:>10.4f}"

            line += f"{np.mean(values):>10.4f}"
            line += f"{np.std(values):>10.4f}"

            f.write(line + "\n")
    
