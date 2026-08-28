from torch.utils.data import Dataset
from utils.util import create_neuron_structure
from utils.config import TREATMENT_MAP, REGION_MAP, PatchConfig
import numpy as np
import torch
import random
from patchify import patchify
import albumentations as A
import scipy.ndimage as ndi
from collections import defaultdict
import pandas as pd

class BasePatchDataset(Dataset):
    def __init__(self, dataset, config: PatchConfig, train=True, augment=False, aug_prob=0.1):
        self.dataset = dataset
        self.train = train
        self.config = config
        self.augment = augment
        self.augment_prob = aug_prob

        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
        ], p=self.augment_prob)

    def __len__(self):
        return len(self.dataset)

    def _extract_patches(self, image, mask) -> list[tuple]:
        """
        Args:
            image: (H, W, C)
            mask: (H, W)
        Returns:
            patches: list of image-mask tuples
        """
        # more overlap during training, less during validation
        steps = (self.config.patch_size // 2) - 1 if self.train else (self.config.patch_size - 1)

        # get patches
        img_patches = patchify(
            image, 
            patch_size = (self.config.patch_size, self.config.patch_size, 3), 
            step = (steps, steps, 3)
        )
        mask_patches = patchify(
            mask, 
            patch_size = (self.config.patch_size, self.config.patch_size), 
            step=(steps, steps)
        )

        H, W = img_patches.shape[0], img_patches.shape[1]
        
        patches = [(img_patches[i,j,0], mask_patches[i,j]) for i in range(H) for j in range(W)]
        return patches

    def _greedy_patching(self, image, mask):
        """
        image: (H, W, C)
        mask: (H, W)
        """
        H, W = mask.shape

        if (H < self.config.patch_size) or (W < self.config.patch_size):
            return []

        candidates = np.column_stack((
            np.random.randint(0, W - self.config.patch_size, size=self.config.num_candidates),          # x position
            np.random.randint(0, H - self.config.patch_size, size=self.config.num_candidates)           # y position
        ))                                                                                      # (num_candidates, 2)
    
        # start with a random patch
        selected = [candidates[random.randint(0, len(candidates) - 1)]]

        for _ in range(1, self.config.num_patches):
            dists = np.min(
                np.linalg.norm(candidates[:, None] - np.array(selected)[None, :], axis=2), 
                axis=1
            )
            selected.append(candidates[np.argmax(dists)])

        patches = [
            (image[y:y + self.config.patch_size, x:x + self.config.patch_size],
            mask[y:y + self.config.patch_size, x:x + self.config.patch_size])
            for x, y in selected
        ]

        return patches

    def _clean_mask_patches(self, mask, area_threshold=150, border_touch_buffer=2):
        H, W = mask.shape

        cleaned_mask_patch = np.zeros_like(mask, dtype=np.uint16)

        # get all unique instance IDs excluding background
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]

        current_label = 1
        # process each instance
        for id in instance_ids:
            # binary mask for current instance
            binary_mask = mask == id

            # label disconnected fragments within the instance
            labeled_fragments, n_frags = ndi.label(binary_mask)

            for frag_id in range(1, n_frags + 1):
                frag_mask = labeled_fragments == frag_id
                frag_area = frag_mask.sum()

                # check if fragment touches the edge of the patch
                y_coords, x_coords = np.where(frag_mask)
                touches_border = (
                    (y_coords < border_touch_buffer).any() or
                    (y_coords >= H - border_touch_buffer).any() or
                    (x_coords < border_touch_buffer).any() or
                    (x_coords >= W - border_touch_buffer).any()
                )

                # remove this fragment if its too small or touching border
                if (touches_border and (frag_area < area_threshold)) or (frag_area < 20):
                    continue

                cleaned_mask_patch[frag_mask] = current_label
                current_label += 1

        return cleaned_mask_patch

    def _get_patches(self, image, mask, filtered=False):
        patches = self._extract_patches(image, mask)

        if filtered:
            # filter out empty patches (filter mainly for diffusion training, should be off for segmentation training)
            patches = [(i_patch, m_patch) for i_patch, m_patch in patches if (m_patch > 0).any()]

        if self.train:
            if len(patches) > self.config.num_patches:
                # randomly pick patches to train on
                idx = np.random.choice(len(patches), size=self.config.num_patches, replace=False)
                patches = [patches[i] for i in idx]
            return patches
        else:
            # Validate / Test on first n patches
            return patches[:self.config.num_patches]

class NeuronDataset(BasePatchDataset):
    def __init__(self, dataset, config: PatchConfig, train=True, augment=False, aug_prob=0.1, filter_empty_patches=False):
        super().__init__(dataset, config, train=train, augment=augment, aug_prob=aug_prob)
        self.filter_empty = filter_empty_patches

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        mask = sample['mask']                                       # (H, W)
        image = sample['image'].astype(np.float32) / 255.0          # (H,W,C) [0,255] --> [0, 1]

        patches = self._get_patches(image, mask, filtered=self.filter_empty)

        results = []
        for img_patch, mask_patch in patches:

            mask_patch = self._clean_mask_patches(mask_patch)
            if self.train and self.augment:
                augmented = self.transforms(image=img_patch, mask=mask_patch.astype(np.uint16))
                img_patch, mask_patch = augmented['image'], augmented['mask']

            img_patch = (img_patch * 2) - 1                                             # range [-1, 1]
            img_patch = np.transpose(img_patch, (2, 0, 1))                              # (H, W, C) --> (C, H, W)

            neuron_structure_patch = create_neuron_structure(mask_patch)
            neuron_structure_patch[0] = (neuron_structure_patch[0] * 2) - 1                                    # [0, 1] --> [-1, 1]

            results.append({
                'image': torch.tensor(img_patch, dtype=torch.float32),                                         # (C, H, W) [-1, 1]
                'neuron_structure': torch.tensor(neuron_structure_patch),                                      # (3, H, W) [-1, 1]
                'treatment_label': torch.tensor(TREATMENT_MAP[sample['treatment']], dtype=torch.long),
                'region_label':torch.tensor(REGION_MAP[sample['region']], dtype=torch.long)
            })
        return results

class NeuronPatchDataset(Dataset):
    def __init__(self, patch_ds):
        super().__init__()
        self.patch_ds = patch_ds

    def __len__(self):
        return len(self.patch_ds)

    def __getitem__(self, idx):
        sample = self.patch_ds[idx]
        mask_patch = sample['mask']
        img_patch = sample['image']

        neuron_structure = create_neuron_structure(mask_patch)
        neuron_structure[0] = (neuron_structure[0] * 2) - 1

        return {
            "image": torch.tensor(img_patch, dtype=torch.float32),
            "neuron_structure": torch.tensor(neuron_structure),
            "treatment_label": torch.tensor(TREATMENT_MAP[sample['treatment']], dtype=torch.long),
            "region_label": torch.tensor(REGION_MAP[sample['region']], dtype=torch.long),
        }



class PatchDataset(Dataset):
    def __init__(self, patch_ds, train=True, use_albu=False, augment_prob=0.25):
        super().__init__()
        self.patch_ds = patch_ds
        self.use_albu = use_albu
        self.train = train

        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Affine(rotate=(-10, 10), scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=0.5),
        ], p=augment_prob)

    def __len__(self):
        return len(self.patch_ds)

    def _build_ground_truth(self, mask_patch, idx):
            H, W = mask_patch.shape
            instance_ids = np.unique(mask_patch)
            instance_ids = instance_ids[instance_ids != 0]
    
            masks, boxes, labels, areas = [], [], [], []
    
            for instance_id in instance_ids:
                binary_mask = (mask_patch == instance_id).astype(np.uint8)
                area = float(np.sum(binary_mask))
    
                # skip invalid mask
                if area == 0:
                    continue
    
                # compute bounding box
                y_indices, x_indices = np.where(binary_mask)
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
    
                # skip invalid boxes
                if x_max <= x_min or y_max <= y_min:
                    continue
    
                masks.append(binary_mask)
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(1)
                areas.append(area)
    
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
                "masks": torch.tensor(np.stack(masks), dtype=torch.uint8) if masks else torch.zeros((0, H, W), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "areas": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros(len(labels), dtype=torch.int64)
            }
    
            return target

    def __getitem__(self, idx):
        sample = self.patch_ds[idx]
        img_patch, mask_patch = sample  

        # augment image + mask
        if self.train and self.use_albu:
            augmented = self.transforms(image=img_patch, mask=mask_patch.astype(np.uint16))
            img_patch, mask_patch = augmented['image'], augmented['mask']

        img_patch = np.transpose(img_patch, (2, 0, 1))
        target = self._build_ground_truth(mask_patch, idx)
        return {
            'image': torch.tensor(img_patch, dtype=torch.float32),
            'target': target
        }

class PatchDatasetComparison(Dataset):
    def __init__(self, patch_ds):
        super().__init__()
        self.patch_ds = patch_ds

    def _calc_cell_density_for_patch(self, mask_patch):
        instance_ids = np.unique(mask_patch)
        instance_ids = instance_ids[instance_ids != 0]

        num_instances = len(instance_ids)

        if num_instances == 0:
            return 0.0 , 0.0

        cell_density = num_instances / (mask_patch.shape[0] * mask_patch.shape[1])
        avg_area_for_patch = np.mean([float(np.sum(mask_patch == instance_id)) for instance_id in instance_ids])

        return cell_density, avg_area_for_patch

    def _calc_background_stats_for_patch(self, image_patch, mask_patch):
        background_pixels = image_patch[mask_patch ==  0]
        return np.mean(background_pixels, axis=0), np.std(background_pixels, axis=0)

    def _calc_stain_intensity_instances(self, image_patch, mask_patch):
        pixels = image_patch[mask_patch != 0]
        if pixels.size == 0:
            n_channels = image_patch.shape[-1]
            return np.full(n_channels, np.nan) , np.full(n_channels, np.nan)
        return np.mean(pixels, axis=0), np.std(pixels, axis=0)      


    def calc_patch_stats_df(self):
        results = []
        for patch in self.patch_ds:
            img_patch, mask_patch, treatment, region = patch.values()

            density, area = self._calc_cell_density_for_patch(mask_patch)
            bg_mean, _ = self._calc_background_stats_for_patch(img_patch, mask_patch)
            stain_mean, _ = self._calc_stain_intensity_instances(img_patch, mask_patch)

            bg_r_mean, bg_g_mean, bg_b_mean = bg_mean
            stain_r_mean, stain_g_mean, stain_b_mean = stain_mean

            results.append({
                "Treatment": treatment,
                "Region": region,
                "Cell Density": density,
                "Cell Area": area,
                "BG_R_mean": bg_r_mean,
                "BG_G_mean": bg_g_mean,
                "BG_B_mean": bg_b_mean,
                "Stain_R_mean": stain_r_mean,
                "Stain_G_mean": stain_g_mean,
                "Stain_B_mean": stain_b_mean,
            })

        return pd.DataFrame(results)
        