import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_scores(scores: dict[str, list], save_path=None):
    kid_scores = scores["kid_scores"]

    # scores evaluated every 25 epochs
    epochs = list(range(25, len(kid_scores) * 25 + 1, 25))

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(epochs, kid_scores, marker="o",linestyle="-", label="KID score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KID score")
    ax.set_title("KID score during training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metrics(metrics: dict[str, list], save_path=None):
    """
    Create metric plot from training results (losses, epoch time, learning rates)
    """

    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    learning_rates = metrics['learning_rates']
    epoch_times = metrics['epoch_times']

    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    ax1, ax2, ax3, ax4 = axes
    
    # Training & Val Loss
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(epochs, train_losses, label="Train Loss", linestyle="-", color='blue')
    ax1.plot(epochs, val_losses, label="Validation Loss", linestyle="-", color='orange')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Loss vs Epochs")

    # Log Loss
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Log(Loss)")
    ax2.plot(epochs, np.log(train_losses), label="Train Loss", linestyle="-", color='blue')
    ax2.plot(epochs, np.log(val_losses), label="Validation Loss", linestyle="-", color='orange')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Log(Loss) vs Epochs")

    # Epoch time
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Time s")
    ax3.plot(epochs, epoch_times, label="Epoch Time", linestyle="-")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Time per Epoch")

    # Learning rate
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("LR")
    ax4.plot(epochs, learning_rates, label="Learning Rate", linestyle="-")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title("Learning Rate vs Epochs")
    
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_samples(samples, treatments, regions, epoch, save_path):
    samples = samples.cpu().numpy()
    B, C, _, _ = samples.shape

    fig, axes = plt.subplots(B, C, figsize=(C * 4, B * 4))

    for i in range(B):
        # Column 1: Binary map
        axes[i, 0].imshow(samples[i, 0])
        axes[i, 0].set_title(f"Binary Map {treatments[i]}-{regions[i]}")
        axes[i, 0].axis("off")

        # Column 2: Horizontal Distance map
        axes[i, 1].imshow(samples[i, 1])
        axes[i, 1].set_title(f"Horizontal Distance Map {treatments[i]}-{regions[i]}")
        axes[i, 1].axis("off")

        # Column 3: Vertical Distance map
        axes[i, 2].imshow(samples[i, 2])
        axes[i, 2].set_title(f"Vertical Distance Map {treatments[i]}-{regions[i]}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(f'{save_path}/epoch_{epoch+1}.png')
    plt.close(fig)

def save_image_samples(images, neurons, samples, epoch, save_path):
    images = images.cpu().numpy()
    neurons = neurons.cpu().numpy()
    samples = samples.cpu().numpy()

    # convert images to [0, 1] range
    images = (images + 1) / 2

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    for i in range(4):

        # Column 1: Images
        axes[i, 0].imshow(np.transpose(images[i], (1, 2, 0)))
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        # Column 2: Neuron structure binary map
        axes[i, 1].imshow(neurons[i,0])
        axes[i, 1].set_title("Neuron Binary Map")
        axes[i, 1].axis("off")

        # Column 3: Generated image
        axes[i, 2].imshow(np.transpose(samples[i], (1, 2, 0)))
        axes[i, 2].set_title("Generated")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_path}/epoch_{epoch+1}.png")
    plt.close(fig)

def plot_metrics_segm(train_losses, val_losses, val_maps, epoch_times, lrs, save_path="training_plot.png"):
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
    
    # mAP
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")
    ax2.plot(epochs, val_maps, label="Val mAP", linestyle="-")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Validation mAP over Epochs")

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

def visualise_target(image, target):

    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    boxes = target['boxes'].cpu().numpy()
    masks = target['masks'].cpu().numpy()

    overlay = np.zeros_like(image)
    alpha = 0.3

    # add each instance mask onto the overy
    for mask in masks:
        color = np.random.rand(3)
        for c in range(3):
            overlay[:, :, c] += mask * color[c]

    blended = ((1 - alpha) * image) + (alpha * overlay)
    blended = np.clip(blended, 0, 1)
    image_with_boxes = (blended * 255).astype(np.uint8).copy()

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        color = tuple(np.random.randint(0, 256, size=3).tolist())
        cv2.rectangle(image_with_boxes, (x_min,y_min), (x_max, y_max), color, 1)

    return image_with_boxes