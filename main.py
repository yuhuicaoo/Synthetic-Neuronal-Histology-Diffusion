import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['HF_HOME'] = '/eresearch/hie/ycao891/Yuhui/hf_cache'
os.environ['TORCH_HOME'] = '/eresearch/hie/ycao891/Yuhui/torch_cache'

from utils.config import TREATMENT_GROUPS, REGION_GROUPS, PatchConfig
from data.load_data import load_image_mask_pairs, split_dataset
from pathlib import Path
from utils.util import collate_fn
from data.create_dataset import NeuronDataset
from torch.utils.data import DataLoader
from training.train_mask import train_model
from training.train_image import train_image_model
import pandas as pd
from utils.plotting import plot_metrics, plot_scores

if __name__ == "__main__":
    ds_root = Path("Neuron Dataset")
    save_dir = Path("saves/mask_training_2")
    os.makedirs(save_dir, exist_ok=True)

    # load data
    base_ds = load_image_mask_pairs(ds_root, TREATMENT_GROUPS, REGION_GROUPS)
    # split data
    train_ds, val_ds, test_ds = split_dataset(base_ds, test_size=0.2, val_size=0.1, seed=42)
    config = PatchConfig()

    # dataloaders
    mask_train_loader = DataLoader(
        NeuronDataset(train_ds, config=config, train=True, augment=True, aug_prob=0.25),
        batch_size = 2, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True,
        collate_fn=collate_fn
    )

    mask_val_loader = DataLoader(
        NeuronDataset(val_ds, config=config, train=False, augment=False), 
        batch_size = 2, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    # image_train_loader = DataLoader(
    #     NeuronDataset(train_ds, config=config, train=True, augment=True, aug_prob=0.25), 
    #     batch_size = 2, 
    #     shuffle=True, 
    #     num_workers = 4, 
    #     pin_memory=True, 
    #     persistent_workers=True,
    #     collate_fn=collate_fn
    # )

    # image_val_loader = DataLoader(
    #     NeuronDataset(val_ds, config=config, train=False, augment=False), 
    #     batch_size = 2, 
    #     shuffle=False, 
    #     num_workers = 4, 
    #     pin_memory=True, 
    #     persistent_workers=True,
    #     collate_fn=collate_fn
    # )
    
    results = train_model(mask_train_loader, mask_val_loader, save_dir=save_dir)
    df = pd.DataFrame(results)
    df.to_csv(f'{save_dir}/results.csv', index=False)
    plot_metrics(results, save_path=f'{save_dir}/training_plot_model.png')

    # results = train_image_model(image_train_loader, image_val_loader, save_dir=save_dir)
    # metrics = {k:v for k,v in results.items() if k != "kid_scores"}
    # scores = {k:v for k,v in results.items() if k == 'kid_scores'}
    # df = pd.DataFrame(metrics)
    # df.to_csv(f'{save_dir}/results.csv', index=False)
    # plot_metrics(metrics, save_path=f'{save_dir}/training_plot_.png')
    # plot_scores(scores, save_path=f'{save_dir}/scores_plot_.png')

