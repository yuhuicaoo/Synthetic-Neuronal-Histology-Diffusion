from pathlib import Path
from tqdm import tqdm
import tifffile as tiff
from sklearn.model_selection import train_test_split, StratifiedKFold
import random
from collections import defaultdict


def load_image_mask_pairs(root: Path, treatment_groups: list[str], region_groups: list[str]):
    """"""
    pairs = []

    for treatment in treatment_groups:
        for region in region_groups:
            folder = root / treatment / region
            print(f"Loading data from: {folder}")

            if not folder.exists():
                continue

            imgs = [f for f in folder.iterdir() if f.name.endswith("_img.tif")]
            for img_path in tqdm(imgs, desc=f"Processing {folder}" , leave=False):
                mask_path = Path(str(img_path).replace("_img", "_masks"))
                
                # check if both image & mask exist
                if not (img_path.exists() and mask_path.exists()):
                    continue
                
                image = tiff.imread(img_path)
                mask = tiff.imread(mask_path)

                pairs.append({
                    "image": image,         # (H, W, C)
                    "mask": mask,           # (H, W)
                    "treatment": treatment,
                    "region": region,
                })

    print(f"Loaded {len(pairs)} image/mask pairs\n")
    return pairs

def split_dataset(base_ds: list[dict], test_size: float = 0.2, val_size: float = 0.1, seed: int = 42):
    """
    Split dataset, ensuring each set is equally distributed in terms of labels.
    """
    stratify_labels = [f"{d['treatment']}_{d['region']}" for d in base_ds]

    trainval_pairs, test_pairs = train_test_split(
        base_ds,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels
    )
    
    rel_val_size = val_size / (1 - test_size)
    trainval_labels = [f"{d['treatment']}_{d['region']}" for d in trainval_pairs]

    train_pairs, val_pairs = train_test_split(
        trainval_pairs,
        test_size    = rel_val_size,
        random_state = seed,
        stratify     = trainval_labels
    )

    train_pairs = list(train_pairs)
    test_pairs  = list(test_pairs)
    val_pairs = list(val_pairs)
    
    print(f"Train size: {len(train_pairs)} | Val size: {len(val_pairs)} | Test size: {len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs

def get_split_for_fold(base_ds: list[dict], folds: int = 5, test_size: float = 0.2, seed: int = 42):
    stratify_labels = [f"{d['treatment']}_{d['region']}" for d in base_ds]

    trainval_pairs, test_pairs = train_test_split(
            base_ds,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_labels
    )

    trainval_labels = [f"{d['treatment']}_{d['region']}" for d in trainval_pairs]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(trainval_pairs, trainval_labels)):
        train_fold = [trainval_pairs[i] for i in train_idx]
        val_fold = [trainval_pairs[i] for i in val_idx]
        folds.append((train_fold, val_fold))
        print(f"Fold {fold_idx}: train={len(train_fold)} | val={len(val_fold)}")

    print(f"Test set: {len(test_pairs)}")
    return folds, test_pairs

def get_subsample_of_dataset(dataset, frac, seed=42):
    rng = random.Random(seed)
    n = max(1, int(len(dataset) * frac))
    selected_indices = rng.sample(range(len(dataset)), k=n)
    return selected_indices