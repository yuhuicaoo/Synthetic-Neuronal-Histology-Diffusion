from utils.config import SegmentationTrainingConfig
from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import time
import math
import torch
from utils.plotting import plot_metrics_segm

class EarlyStopper():
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_mAP = -float('inf')
        self.early_stop = False

    def __call__(self, val_mAP):
        if val_mAP > (self.best_mAP + self.delta):
            self.best_mAP = val_mAP
            self.counter = 0
            return False

        self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    """
    Create a schedule with a warmup period followed by cosine annealing.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def evaluate_segmentation_model(model, test_loader, device):
    model.to(device)
    model.eval()

    metric = MeanAveragePrecision(iou_type='segm', box_format='xyxy')

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Eval', unit='batch', leave=False):
            images, targets = batch['image'], batch['target']
            if len(images) == 0:
                continue

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

            with autocast('cuda'):
                preds = model(images)
            for p in preds:
                p['masks'] = (p['masks'] > 0.5).squeeze(1).byte()

            metric.update(preds, targets)

    results = metric.compute()
    tqdm.write(
        f"mAP: {results['map']:.4f} | "
        f"mAP@0.50: {results['map_50']:.4f} | "
        f"mAP@0.75: {results['map_75']:.4f} | "
        f"mAR@100: {results['mar_100']:.4f}"
    )

    return {
        'map': results['map'].item(),
        'map_50': results['map_50'].item(),
        'map_75': results['map_75'].item(),
        'mar_100': results['mar_100'].item()
    }


def run_segmentation_training(model, train_loader, val_loader, save_dir, fold=0):
    save_path = f'{save_dir}/resnet_maskRCNN_fold{fold}.pth'
    config = SegmentationTrainingConfig()

    model.to(config.device)

    optimiser = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    earlystopper = EarlyStopper(patience=15, delta=1e-4)

    steps_per_epoch = len(train_loader)
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimiser,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.01
    )

    scaler = GradScaler('cuda')

    train_losses, val_losses, val_maps = [], [] ,[]
    epoch_times, lrs = [] , []
    best_val_map = 0
    val_map_metric = MeanAveragePrecision(iou_type='segm')

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc='Train', unit='batch', leave=False):
            images, targets = batch['image'], batch['target']
            if len(images) == 0:
                continue

            images = [img.to(config.device) for img in images]
            targets = [{k: v.to(config.device) for k, v in t.items()} for t in targets]

            optimiser.zero_grad()

            with autocast('cuda'):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loss
        model.train()       # stay in train mode to get losses
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Val (loss)', unit='batch', leave=False):
                images, targets = batch['image'], batch['target']
                if len(images) == 0:
                    continue

                images = [img.to(config.device) for img in images]
                targets = [{k: v.to(config.device) for k,v in t.items()} for t in targets]

                with autocast('cuda'):
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Validation mAP
        model.eval()
        val_map_metric.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Val (mAP)', unit='batch', leave=False):
                images, targets = batch['image'], batch['target']
                if len(images) == 0:
                    continue

                images = [img.to(config.device) for img in images]
                targets = [{k: v.to(config.device) for k,v in t.items()} for t in targets]

                with autocast('cuda'):
                    preds = model(images)
                for p in preds:
                    p['masks'] = (p['masks'] > 0.5).squeeze(1).byte()

                val_map_metric.update(preds, targets)

        metrics = val_map_metric.compute()
        val_map = metrics['map'].item()
        val_maps.append(val_map)

        tqdm.write(
            f"Epoch [{epoch+1}/{config.num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val mAP: {val_map:.4f} | "
            f"mAP@0.50: {metrics['map_50']:.4f} | "
            f"mAP@0.75: {metrics['map_75']:.4f} | "
            f"mAR@100: {metrics['mar_100']:.4f}"
        )

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        lrs.append(optimiser.param_groups[0]['lr'])
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        print(f"Epoch Time: {epoch_time:.2f}s | Avg Epoch Time: {avg_epoch_time:.2f}s")

        # Save best model
        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model (mAP: {best_val_map:.4f}) to {save_path}")

        if earlystopper(val_map):
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Training Complete! Best Val mAP: {best_val_map:.4f}")
    print(f"Average Epoch Time: {sum(epoch_times) / len(epoch_times):.2f}s")
    plot_metrics_segm(train_losses, val_losses, val_maps, epoch_times, lrs, save_path=f'{save_dir}/training_plot_fold{fold}.png')

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maps': val_maps,
        'lrs': lrs,
        'epoch_times': epoch_times,
    }, best_val_map