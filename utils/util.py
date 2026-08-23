import numpy as np
from skimage.measure import regionprops
from skimage import filters, morphology, measure
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
import torch
from utils.config import PatchConfig
from patchify import patchify
from tqdm import tqdm

def create_neuron_structure(mask):
    H, W = mask.shape

    # create binary map where, 0 = background and 1 = neuron
    binary_mask = (mask > 0 ).astype(np.float32)        # (H, W) range [0, 1]

    # create horizontal & vertical distance maps
    hover_h = np.zeros((H, W), dtype=np.float32)        # (H, W)
    hover_v = np.zeros((H, W), dtype=np.float32)        # (H, W)

    for prop in regionprops(mask):
        # get center of mass for this neuron instance
        center_y, center_x = prop.centroid

        # get the coords for this instance
        coords = prop.coords
        coords_y, coords_x = coords[:,0], coords[:,1]

        # horizontal distance from the center of the neuron instance
        x_dist = coords_x - center_x
        neg = x_dist < 0
        pos = x_dist > 0

        if np.any(x_dist < 0):
            x_dist[neg] /= -x_dist[neg].min()

        if np.any(pos):
            x_dist[pos] /= x_dist[pos].max()
        
        # vertical distance from the center of the neuron instance
        y_dist = coords_y - center_y
        neg = y_dist < 0
        pos = y_dist > 0

        if np.any(neg):
            y_dist[neg] /= -y_dist[neg].min()

        if np.any(pos):
            y_dist[pos] /= y_dist[pos].max()

        hover_h[coords_y, coords_x] = x_dist
        hover_v[coords_y, coords_x] = y_dist

    # stack binary_map and distance maps into a 3-channel structure 
    neuron_structure = np.stack([binary_mask, hover_h, hover_v], axis=0)        # (3, H , W)
    return neuron_structure

def cosine_beta_schedule(timesteps, s=0.008):
    t = torch.linspace(0, timesteps, timesteps+1) / timesteps
    
    # Cumulative noise (alpha_bar) follows a cosine curve
    alpha_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]  # normalize to start at 1
    
    # Derive betas from alpha_bar
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, min=1e-4, max=0.999)

def watershed_postprocess(neuron_structure, fg_threshold = 0.5, hover_threshold = 0.4,min_area = 25):

    binary_map, horizontal_map, vertical_map = neuron_structure

    fg_mask = binary_map > fg_threshold

    horizontal_map = gaussian_filter(horizontal_map, sigma=0.5)
    vertical_map   = gaussian_filter(vertical_map, sigma=0.5)

    grad_h , grad_v = filters.sobel_h(horizontal_map), filters.sobel_v(vertical_map)

    energy_landscape = np.sqrt(grad_h**2 + grad_v**2)
    energy_landscape_normalised = (energy_landscape - energy_landscape.min()) / (energy_landscape.max() - energy_landscape.min() + 1e-8)

    dist = np.sqrt(horizontal_map**2 + vertical_map**2)
    seed_mask = dist < hover_threshold
    seed_mask &= (binary_map > 0.5)

    seed_mask = morphology.binary_opening(seed_mask, morphology.disk(2))
    seed_mask = morphology.binary_closing(seed_mask, morphology.disk(2))
    seed_mask = morphology.remove_small_objects(seed_mask, min_size=min_area)

    markers = measure.label(seed_mask)
    for region in measure.regionprops(markers):
        if region.area < min_area:
            markers[markers == region.label] = 0

    markers = measure.label(markers > 0)

    instance_map = watershed(
        energy_landscape_normalised,
        markers=markers,
        mask=fg_mask
    )

    return instance_map

def collate_fn(batch):
    """Flattens [ [patch1, patch2, ...], [patch1, patch2, ...], ... ] to one batch list"""
    flat_batch = []
    for sample_group in batch:
        flat_batch.extend(sample_group)

    collated = {}
    for key in flat_batch[0]:
        values = [sample[key] for sample in flat_batch]

        if isinstance(values[0], torch.Tensor) and all(v.shape == values[0].shape for v in values):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values

    return collated

def collate_fn_segm(batch):
    images, targets = [], []

    for sample in batch:
        images.append(sample['image'])
        targets.append(sample['target'])

    return {
        "image": torch.stack(images),
        "target": targets
    }

def get_patches(dataset, config:PatchConfig):

    patches = []

    for sample in dataset:
        image = sample['image'].astype(np.float32) / 255.0
        mask = sample['mask']

        # patchify
        img_patches = patchify(
            image, 
            patch_size=(config.patch_size, config.patch_size, 3),
            step=(config.patch_size, config.patch_size, 3)
        )
        mask_patches = patchify(
            mask,
            patch_size=(config.patch_size, config.patch_size),
            step=(config.patch_size, config.patch_size)
        )

        H,W = img_patches.shape[:2]

        curr_patches = [(img_patches[i,j,0], mask_patches[i,j]) for i in range(H) for j in range(W)]
        patches.extend(curr_patches[:config.num_patches])

    return patches

def get_patches_2(dataset, config:PatchConfig, filter_empty=False):

    patches = []

    for sample in tqdm(dataset, desc=f"Patching", leave=False):
        image = sample['image']
        mask = sample['mask']
        treatment = sample['treatment']
        region = sample['region']

        # patchify
        img_patches = patchify(
            image, 
            patch_size=(config.patch_size, config.patch_size, 3),
            step=(config.stride, config.stride, 3)
        )
        mask_patches = patchify(
            mask,
            patch_size=(config.patch_size, config.patch_size),
            step=(config.stride, config.stride)
        )

        H,W = img_patches.shape[:2]

        curr_patches = []
        for i in range(H):
            for j in range(W):
                mask_patch = mask_patches[i, j]

                if filter_empty:
                    n_instances = len(np.unique(mask_patch)) - 1
                    if n_instances < 1:
                        continue

                curr_patches.append({
                    "image": img_patches[i,j,0].astype(np.float32) / 255.0,
                    "mask": mask_patch,
                    "treatment": treatment,
                    "region": region,
                })
        patches.extend(curr_patches[:config.num_patches])
    print(len(patches))
    return patches