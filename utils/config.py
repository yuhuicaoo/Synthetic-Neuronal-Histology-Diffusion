from dataclasses import dataclass
import torch

@dataclass
class DiffusionConfig:
    image_size: int = 256       # patch_size
    in_channels: int = 3        # nuclei structure has 3 channels, RGB image also has 3 channels
    base_channels: int = 256    # network width (larger = smarter but slower)
    time_emb_dim: int = 256     # size of time signal vector
    time_steps: int = 1000      # how many steps until image becomes pure static noise
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # noise scheduler (small noise at start, more noise at end)
    beta_start: float = 1e-4
    beta_end: float = 0.02
    num_regions: int = 8        # CA3, CA4, CA12, DG, PS1, PS2, TH1, TH2
    num_treatments: int = 3     # HI, HI+HYPO, SHAM
    epochs: int = 500
    learning_rate: float = 1e-4

@dataclass
class PatchConfig:
    patch_size: int = 256
    num_patches: int = 12
    num_candidates: int = 2000
    stride: int = 128

@dataclass
class SegmentationTrainingConfig:
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    warmup_epochs: int = 5

@dataclass
class AttentionConfig:
    num_heads: int = 8
    num_embd: int = 1024
    dropout_rate: float = 0.1

TREATMENT_GROUPS = ['HI', 'HI + HYPO', 'SHAM']
REGION_GROUPS = ['CA3', 'CA4', 'CA12', 'DG', 'PS1', 'PS2', 'TH1', 'TH2']

TREATMENT_MAP = {t: i for i, t in enumerate(TREATMENT_GROUPS)}
REGION_MAP = {r: i for i, r in enumerate(REGION_GROUPS)}