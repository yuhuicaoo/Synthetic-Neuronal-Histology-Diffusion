import torch
import time
from model.controlnetImage_diffusion import ControlNetImageDiffusion
from utils.config import DiffusionConfig
from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from utils.plotting import save_image_samples
from training.training_funcs import sample_with_ema, evaluate_kid, EMA

def train_image_model(train_loader, val_loader, save_dir):
    torch.manual_seed(seed=42)

    config = DiffusionConfig()
    diffusion_model = ControlNetImageDiffusion(
        config = config,
        pretrained_model= 'sd-legacy/stable-diffusion-v1-5',
        train_unet = False,
        unfreeze_last_n_down_blocks = 0,
        use_unet_lora = False,
        lora_rank = 4
    ).to(config.device)

    ema_decay = 0.999
    ema_controlnet = EMA(diffusion_model.controlnet, decay=ema_decay)
    ema_label_conditioner = EMA(diffusion_model.label_conditioner, decay=ema_decay)

    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch
    effective_window = 1 / (1 - ema_decay)
    print(f"steps/epoch: {steps_per_epoch} | total steps: {total_steps} | EMA window: {effective_window:.0f} | ratio (window/total): {effective_window/total_steps:.3f}")

    trainable_parameters = [p for p in diffusion_model.parameters() if p.requires_grad]
    optimiser = AdamW(trainable_parameters, lr=config.learning_rate, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max = config.epochs, eta_min=1e-6)
    scaler = GradScaler(device='cuda')

    train_losses, val_losses, lrs, epoch_times = [], [], [], []
    kid_scores = []
    best_kid_score = float('inf')

    # get fixed sample for visualisations
    sample_batch = next(iter(val_loader))
    sample_neurons = sample_batch['neuron_structure'][:4].to(config.device)
    sample_images = sample_batch['image'][:4].to(config.device)
    sample_treatments = sample_batch['treatment_label'][:4].to(config.device)
    sample_regions = sample_batch['region_label'][:4].to(config.device)

    for epoch in range(config.epochs):
        start = time.time()
        diffusion_model.train()
        train_loss = 0.0

        # Training
        for batch in tqdm(train_loader, desc="Train", unit="batch", leave=False):
            x0 = batch['image'].to(config.device)
            neuron_structure = batch['neuron_structure'].to(config.device)
            treatment_label = batch['treatment_label'].to(config.device)
            region_label = batch['region_label'].to(config.device)

            optimiser.zero_grad(set_to_none=True)

            with autocast('cuda'):
                loss = diffusion_model(x0, neuron_structure, treatment_label, region_label, cond_drop_prob=0.1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            ema_controlnet.update(diffusion_model.controlnet)
            ema_label_conditioner.update(diffusion_model.label_conditioner)

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        diffusion_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", unit="batch", leave=False):
                x0 = batch['image'].to(config.device)
                neuron_structure = batch['neuron_structure'].to(config.device)
                treatment_label = batch['treatment_label'].to(config.device)
                region_label = batch['region_label'].to(config.device)
                

                with autocast('cuda'):
                    loss = diffusion_model(x0, neuron_structure, treatment_label, region_label, cond_drop_prob = 0.0)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        lrs.append(current_lr)
        tqdm.write(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.2f}s")

        # generate samples
        if (epoch + 1) % 25  == 0:
            samples = sample_with_ema(
                diffusion_model,
                ema_controlnet, 
                ema_label_conditioner, 
                neuron_structure=sample_neurons, 
                treatment_label=sample_treatments,
                region_label=sample_regions,
                guidance_scale=2.0
            )
            save_image_samples(sample_images, sample_neurons, samples, epoch, save_dir)
            print(f"Saved sample images to: {save_dir}")

            print(f"Calculating Kernel Inception Distance (KID)")
            kid_mean, kid_std = evaluate_kid(
                diffusion_model=diffusion_model,
                ema_controlnet=ema_controlnet,
                ema_label_conditioner=ema_label_conditioner,
                loader=val_loader, 
                device=config.device,
                guidance_scale=2.0
            )

            kid_scores.append(kid_mean)
            print(f"KID score: {kid_mean:.5f} ± {kid_std:.5f}")

            # save best model
            if kid_mean < best_kid_score:
                best_kid_score = kid_mean
                torch.save({
                    'controlnet': diffusion_model.controlnet.state_dict(),
                    'label_conditioner': diffusion_model.label_conditioner.state_dict(),
                    'ema_controlnet': ema_controlnet.state_dict(),
                    'ema_label_conditioner': ema_label_conditioner.state_dict(),
                    # 'unet_lora': {k: v for k, v in diffusion_model.unet.state_dict().items() if 'lora' in k}
                },  f'{save_dir}/image_best.pth')
                print(f"Saved new best model (KID score: {best_kid_score:.4f}) to {save_dir}/image_best.pth")

    # save model at the end of training 
    torch.save({
        'controlnet': diffusion_model.controlnet.state_dict(),
        'label_conditioner': diffusion_model.label_conditioner.state_dict(),
        'ema_controlnet': ema_controlnet.state_dict(),
        'ema_label_conditioner': ema_label_conditioner.state_dict(),
        # 'unet_lora': {k: v for k, v in diffusion_model.unet.state_dict().items() if 'lora' in k}
    },  f'{save_dir}/image_epoch{config.epochs}.pth')

    print(f"Saved model to {save_dir}/image_epoch{config.epochs}.pth")
    print(f"\nTraining Complete! Best KID score: {best_kid_score:.4f}")
    print(f"Average Epoch Time: {sum(epoch_times) / len(epoch_times):.2f}s")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': lrs,
        'epoch_times': epoch_times,
        'kid_scores': kid_scores,
    }