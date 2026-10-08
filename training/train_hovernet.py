from utils.config import SegmentationTrainingConfig
from tqdm import tqdm
from torch.optim import AdamW, lr_scheduler
from torch.amp import GradScaler, autocast
import time
import numpy as np
import torch
from utils.util import count_params
from training.train_segmentation import EarlyStopper, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from monai.apps.pathology.losses.hovernet_loss import HoVerNetLoss
from monai.utils.enums import HoVerNetBranch
from monai.metrics import DiceMetric
from monai.apps.pathology.transforms import HoVerNetInstanceMapPostProcessing
from model.hovernet import create_hovernet_model
from monai.data import decollate_batch
from monai.transforms import (
    Compose,
    AsDiscrete,
    Activations
)

pretrained_model = "https://drive.google.com/u/1/uc?id=1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5&export=download"


class PQMeter:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.tp = 0 
        self.fp = 0
        self.fn = 0
        self.iou_sum = 0.0

    def update(self, pred, true):
        pred = np.asarray(pred).astype(np.int64)
        true = np.asarray(true).astype(np.int64)

        # get instance ids
        pred_ids = np.unique(pred)
        true_ids = np.unique(true)

        # remove background id
        pred_ids = pred_ids[pred_ids != 0]
        true_ids = true_ids[true_ids != 0]

        num_pred = len(pred_ids)
        num_true = len(true_ids)

        if num_pred == 0 and num_true == 0:
            return

        if num_pred == 0:
            self.fn += num_true
            return

        if num_true == 0:
            self.fp += num_pred
            return

        # create IoU Matrix
        iou = np.zeros((num_pred, num_true), dtype=np.float32)

        for i , pred_id in enumerate(pred_ids):
            pred_mask = (pred == pred_id)

            for j , true_id in enumerate(true_ids):
                true_mask = (true == true_id)

                intersection = np.logical_and(pred_mask, true_mask).sum()

                if intersection == 0:
                    continue

                union = np.logical_or(pred_mask, true_mask).sum()
                iou[i, j] = intersection / union

        # find candidates above iou threshold
        candidates = np.argwhere(iou > self.iou_threshold)

        # sort candidates by IoU, descending order
        candidates = sorted(candidates, key = lambda x:iou[x[0], x[1]], reverse = True)

        # one-to-one matching
        matched_pred = set()
        matched_true = set()

        matched_ious = []

        for pred_idx, true_idx in candidates:
            # check if already matched
            if pred_idx in matched_pred:
                continue

            if true_idx in matched_true:
                continue

            # match
            matched_pred.add(pred_idx)
            matched_true.add(true_idx)

            matched_ious.append(iou[pred_idx, true_idx])

        tp = len(matched_ious)
        fp = num_pred - tp
        fn = num_true - tp

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.iou_sum += sum(matched_ious)

    def compute(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        # recongnition quality
        rq = (2*tp) / max(2 * tp + fp + fn, 1e-9)
        # segmentation quality
        sq = self.iou_sum / max(tp, 1)
        # panoptic quality
        pq = rq * sq
        # F1 = RQ
        f1 = rq

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return {
            'pq': pq,
            'rq': rq,
            'sq': sq,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

def plot_metrics_hovernet(train_losses, val_losses, val_pqs, val_f1s, epoch_times, lrs, save_path="training_plot.png"):
    epochs = list(range(1, len(train_losses) + 1))
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 14))
    ax1, ax2, ax3, ax4 = axes
    
    # Losses
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Log(Loss)")
    ax1.plot(epochs, np.log(train_losses), label="Training", linestyle="-")
    ax1.plot(epochs, np.log(val_losses), label="Validation", linestyle="--")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Log(Loss) over Epochs")
    
    # PQ and F1 score
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.plot(epochs, val_pqs, label="Val PQ", linestyle="-")
    ax2.plot(epochs, val_f1s, label="Val F1", linestyle="--")
    best_epoch = int(np.argmax(val_pqs)) + 1
    ax2.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7,
                label=f"Best PQ (epoch {best_epoch}: {max(val_pqs):.3f})")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Validation PQ and F1 over Epochs")

    # Epoch time
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Time s")
    ax3.plot(epochs, epoch_times, label="Epoch Time", linestyle="-")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Time per Epoch including Validation")

    # Learning rate
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("LR")
    ax4.plot(epochs, lrs, label="Learning Rate", linestyle="-")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title("Learning Rate over Epochs")
    
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def post_process_batch_hovernet(preds, post_proc):
    """preds: dict of batched model outputs -> list of (H, W) int instance maps."""
    np_out = preds[HoVerNetBranch.NP.value].detach().cpu()   # (B, 2, H, W)
    hv_out = preds[HoVerNetBranch.HV.value].detach().cpu()   # (B, 2, H, W)

    inst_maps = []
    for np_i, hv_i in zip(np_out, hv_out):                   # per-sample
        _, inst_map  = post_proc(np_i, hv_i)                  # inst_map: (1, H, W)
        if torch.is_tensor(inst_map):
            inst_map = inst_map.cpu().numpy()
        inst_maps.append(np.squeeze(inst_map).astype(np.int64))
    return inst_maps

def evaluate_hovernet(model, test_loader, device):
    model.to(device)
    model.eval()

    pq_meter = PQMeter(iou_threshold=0.5)
    dice_metric = DiceMetric(include_background=False, reduction='mean')

    post_process_np_branch = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])

    post_proc = HoVerNetInstanceMapPostProcessing(
        activation="softmax",
        mask_threshold=0.5,
        min_object_size=10,
        sobel_kernel_size=5,
        marker_threshold=0.4,
        marker_radius=2,
    )

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Eval', unit='batch', leave=False):
            images = batch['image'].to(device)
            binary_map = batch['binary_map'].to(device)

            preds = model(images)

            # compute Dice score
            outputs = [post_process_np_branch(i[HoVerNetBranch.NP.value]) for i in decollate_batch(preds)]
            labels = decollate_batch(binary_map)
            dice_metric(y_pred=outputs, y=labels)

            # convert binary + hover map outputs to instance map
            predicted_instance_map = post_process_batch_hovernet(preds, post_proc)

            gt_instance_map = batch['instance_map'].cpu().numpy()

            for pred, truth in zip(predicted_instance_map, gt_instance_map):
                pq_meter.update(pred, truth)

    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    metrics = pq_meter.compute()

    tqdm.write(
        f"PQ: {metrics['pq']:.4f} | "
        f"F1 / RQ: {metrics['f1']:.4f} | "
        f"SQ: {metrics['sq']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"Dice: {dice_score:.4f}"
    )

    return {
        **metrics,
        'dice': dice_score
    }


def run_hovernet_training(train_loader, val_loader, save_dir, fold=0):
    save_path = f'{save_dir}/HoVerNet_fold{fold}.pth'
    config = SegmentationTrainingConfig()

    model = create_hovernet_model(stage=0, out_classes=0, pretrained_model=pretrained_model, ckpt_path=save_path, device=config.device)
    model.to(config.device)

    total_params, trainable_params = count_params(model)
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    optimiser = AdamW(
        (p for p in model.parameters() if p.requires_grad), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )

    scheduler = lr_scheduler.StepLR(optimiser, step_size=25)

    scaler = GradScaler('cuda')
    loss_function = HoVerNetLoss(lambda_hv_mse=1.0)

    train_losses, val_losses, val_pqs, val_f1s = [], [] ,[], []
    epoch_times, lrs = [] , []
    best_val_pq = -1

    meter = PQMeter(iou_threshold=0.5)
    dice_metric = DiceMetric(include_background=False, reduction='mean')

    post_proc = HoVerNetInstanceMapPostProcessing(
        activation="softmax",
        mask_threshold=0.5,
        min_object_size=10,
        sobel_kernel_size=5,
        marker_threshold=0.4,
        marker_radius=2,
    )

    post_process_np_branch = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])

    for epoch in range(config.num_epochs):
        if epoch == 50:
            model = create_hovernet_model(stage=1, out_classes=0, pretrained_model=None, ckpt_path=save_path, device=config.device)
            optimiser = AdamW(
                    (p for p in model.parameters() if p.requires_grad), 
                    lr=config.lr, 
                    weight_decay=config.weight_decay
                )
            scheduler = lr_scheduler.StepLR(optimiser, step_size=25)
            print(f"Loaded model from {save_path} - stage 2 of training")

            total_params, trainable_params = count_params(model)
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc='Train', unit='batch', leave=False):
            images = batch['image'].to(config.device)
            binary_map = batch['binary_map'].to(config.device)
            hv_map = batch['hv_map'].to(config.device)

            labels = {
                HoVerNetBranch.NP: binary_map,
                HoVerNetBranch.HV: hv_map
            }

            optimiser.zero_grad()

            with autocast('cuda'):
                outputs = model(images)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()


            train_loss += loss.item()

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        meter.reset()
        dice_metric.reset()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Val (loss)', unit='batch', leave=False):
                images = batch['image'].to(config.device)
                binary_map = batch['binary_map'].to(config.device)
                hv_map = batch['hv_map'].to(config.device)

                labels = {
                    HoVerNetBranch.NP: binary_map, 
                    HoVerNetBranch.HV: hv_map,
                }

                with autocast('cuda'):
                    preds = model(images)
                    loss = loss_function(preds, labels)

                val_loss += loss.item()

                # dice score
                val_outputs = [post_process_np_branch(i[HoVerNetBranch.NP.value]) for i in decollate_batch(preds)]
                val_label = decollate_batch(binary_map)
                dice_metric(y_pred=val_outputs, y=val_label)

                # PQ & F1 score
                pred_inst_maps = post_process_batch_hovernet(preds, post_proc)
                gt_instance_map = batch['instance_map'].cpu().numpy()

                for pi, ti in zip(pred_inst_maps, gt_instance_map):
                    meter.update(pi, np.squeeze(ti))

        val_dice = dice_metric.aggregate().item()
        m = meter.compute()

        val_pq, val_f1 = m['pq'], m['f1']
        val_pqs.append(val_pq)
        val_f1s.append(val_f1)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        tqdm.write(
            f"Epoch [{epoch+1}/{config.num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"PQ: {m['pq']:.4f} | F1: {m['f1']:.4f} | SQ: {m['sq']:.4f} | Dice: {val_dice:.4f} | "
            f"Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f}" 
        )

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        lrs.append(optimiser.param_groups[0]['lr'])
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        print(f"Epoch Time: {epoch_time:.2f}s | Avg Epoch Time: {avg_epoch_time:.2f}s")

        # Save best model
        if val_pq > best_val_pq:
            best_val_pq = val_pq
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model (PQ: {best_val_pq:.4f}) to {save_path}")

    print(f"Training Complete! Best Val PQ: {best_val_pq:.4f}")
    print(f"Average Epoch Time: {sum(epoch_times) / len(epoch_times):.2f}s")
    plot_metrics_hovernet(train_losses, val_losses, val_pqs, val_f1s, epoch_times, lrs, save_path=f'{save_dir}/training_plot_fold{fold}.png')

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_pqs': val_pqs,
        'val_f1s': val_f1s,
        'lrs': lrs,
        'epoch_times': epoch_times,
    }, best_val_pq