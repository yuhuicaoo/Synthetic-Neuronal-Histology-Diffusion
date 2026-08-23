import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
os.environ['HF_HOME'] = '/eresearch/hie/ycao891/Yuhui/hf_cache'
os.environ['TORCH_HOME'] = '/eresearch/hie/ycao891/Yuhui/torch_cache'

from data.load_data import load_image_mask_pairs, get_split_for_fold, get_subsample_of_dataset
from pathlib import Path
from utils.util import collate_fn_segm, get_patches
from utils.config import TREATMENT_GROUPS, REGION_GROUPS, PatchConfig
from torch.utils.data import DataLoader, Subset
from data.create_dataset import PatchDataset
from model.swin_maskRCNN import create_maskrcnn_resnet50
from training.train_segmentation import run_segmentation_training
import numpy as np
import torch
import random


if __name__ == "__main__":
    ds_root = Path("Neuron Dataset")
    ds_root_gen = Path("Neuron Dataset Generated 3")

    save_dir = Path('saves/segmentation_training_75_gen_2')
    os.makedirs(save_dir, exist_ok=True)
    
    base_ds = load_image_mask_pairs(ds_root, TREATMENT_GROUPS, REGION_GROUPS)
    generated_ds = load_image_mask_pairs(ds_root_gen, TREATMENT_GROUPS, REGION_GROUPS)
    gen_patches_ds = [(p['image'].astype(np.float32) / 255.0, p['mask']) for p in generated_ds]

    config = PatchConfig()
    
    folds, test_ds = get_split_for_fold(base_ds, folds=5, test_size=0.2, seed=42)

    fold_results, fold_maps = [], []

    aug_prob = 0.25
    subset_frac = 0.75

    for fold_idx, (t_ds, v_ds) in enumerate(folds):
        print(f"\n === Fold {fold_idx} ===")

        fold_seed = 42 + fold_idx
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        random.seed(fold_seed)

        if subset_frac < 1.0:
            train_ds_subset = Subset(t_ds, get_subsample_of_dataset(t_ds, frac=subset_frac, seed=42))
        else:
            train_ds_subset = t_ds

        print(f"100% of original dataset: {len(t_ds)} pairs, {subset_frac*100}% of original dataset: {len(train_ds_subset)} pairs\n")

        train_patches = get_patches(train_ds_subset, config=config) + gen_patches_ds
        val_patches = get_patches(v_ds, config=config)

        print(f"Train patches: {len(train_patches)} | Val patches: {len(val_patches)}")

        train_loader = DataLoader(
            PatchDataset(train_patches, train=True, use_albu=False, augment_prob=aug_prob),
            batch_size = 32, 
            shuffle=True, 
            num_workers = 4, 
            pin_memory=True, 
            persistent_workers=True, 
            collate_fn=collate_fn_segm,
        )
        val_loader = DataLoader(
            PatchDataset(val_patches, train=False, use_albu=False, augment_prob=aug_prob),
            batch_size = 32, 
            shuffle=False, 
            num_workers = 4, 
            pin_memory=True,
            persistent_workers=True, 
            collate_fn=collate_fn_segm,
        )

        model = create_maskrcnn_resnet50(num_classes=2, min_size=256, max_size=256)

        results, best_val_map = run_segmentation_training(
            model, train_loader, val_loader, save_dir=save_dir, fold=fold_idx
        )

        fold_results.append(results)
        fold_maps.append(best_val_map)

        del model
        torch.cuda.empty_cache()

    with open(f"{save_dir}/training_results.txt", "w") as f:
        f.write(f"Resnet Backbone on MaskRCNN training - baseline: {subset_frac*100}% of original dataset\n")
        f.write("Training Results:\n\n\n")
        for fold in range(5):
            f.write(f"\n\n########### FOLD {fold} ###########\n")
            results = fold_results[fold]
            for k, v in results.items():
                f.write(f"{k}:\n")
                if isinstance(v, list):
                    f.write(", ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in v) + "\n\n")

            f.write(f"{fold_maps[fold]:.4f}\n\n")

        f.write(f"Average mAP:{np.mean(fold_maps)}\n")
        f.write(f"STD mAP:{np.std(fold_maps)}\n")