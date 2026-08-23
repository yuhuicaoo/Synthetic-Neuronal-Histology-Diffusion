import torch
import time
from model.mask_diffusion import Mask_Diffusion
from utils.config import DiffusionConfig
from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from utils.plotting import save_samples
from utils.config import TREATMENT_MAP, REGION_MAP, TREATMENT_GROUPS, REGION_GROUPS
import random
import os
from training.training_funcs import EMA

def train_model(train_loader, val_loader, save_dir=None):
    torch.manual_seed(seed=42)
    config = DiffusionConfig()
    diffusion_model = Mask_Diffusion(config=config).to(config.device)
    optimiser = AdamW(diffusion_model.parameters(), lr = config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max = config.epochs, eta_min=1e-6)
    scaler = GradScaler(device='cuda')

    train_losses, val_losses, lrs, epoch_times = [], [], [], []
    best_loss = float('inf')

    ema_decay = 0.999
    ema_model = EMA(model=diffusion_model, decay=ema_decay)

    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch
    effective_window = 1 / (1 - ema_decay)
    print(f"steps/epoch: {steps_per_epoch} | total steps: {total_steps} | EMA window: {effective_window:.0f} | ratio (window/total): {effective_window/total_steps:.3f} \n")

    n_samples = 4
    os.makedirs(save_dir, exist_ok = True)

    for epoch in range(config.epochs):
        start = time.time()
        diffusion_model.train()
        train_loss = 0.0

        # Training
        for batch in tqdm(train_loader, desc=f"Train", unit="batch", leave=False):
            x0 = batch['neuron_structure'].to(config.device)
            t_label = batch['treatment_label'].to(config.device)
            r_label = batch['region_label'].to(config.device)

            optimiser.zero_grad(set_to_none=True)

            with autocast('cuda'):
                loss = diffusion_model(x0, t_label, r_label, cond_drop_prob = 0.1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            ema_model.update(diffusion_model)

            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        diffusion_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc = "Val", unit='batch', leave=False):
                x0 = batch['neuron_structure'].to(config.device)
                t_label = batch['treatment_label'].to(config.device)
                r_label = batch['region_label'].to(config.device)

                with autocast('cuda'):
                    loss = diffusion_model(x0, t_label, r_label, cond_drop_prob = 0.0)
                
                val_loss += loss.item()
        
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        lrs.append(current_lr)
        tqdm.write(
            f"Epoch [{epoch+1}/{config.epochs}] | "
            f"Train Loss: {avg_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f} | " 
            f"LR: {current_lr:.2e} | " 
            f"Time: {epoch_time:.2f}s"
        )

        # save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'model': diffusion_model.state_dict(),
                'ema_model': ema_model.state_dict(),
            }, f'{save_dir}/model_best.pth')
            print(f"Saved new best model (loss: {best_loss:.5f}) to {save_dir}/model_best.pth'")
        
        if (epoch + 1) % 25 == 0:
            treatments = random.choices(TREATMENT_GROUPS, k = n_samples)
            regions = random.choices(REGION_GROUPS, k = n_samples)

            treatment_labels = torch.tensor([TREATMENT_MAP[t] for t in treatments]).to(config.device)
            region_labels = torch.tensor([REGION_MAP[r] for r in regions]).to(config.device)

            samples = ema_model.ema_model.sample(n_samples, treatment_labels, region_labels, guidance_scale=2.0)
            save_samples(samples, treatments, regions, epoch, save_path=save_dir)

    # save model at the end
    torch.save({
        'model': diffusion_model.state_dict(),
        'ema_model': ema_model.state_dict(),
    }, f'{save_dir}/model_epoch{config.epochs}.pth')

    print(f"Saved model to '{save_dir}/model_epoch{config.epochs}")
    print(f"\nTraining Complete - Best loss: {best_loss:.5f}")
    print(f"Average Epoch Time: {sum(epoch_times) / len(epoch_times):.2f}s")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': lrs,
        'epoch_times': epoch_times
    }