"""
Dataset Inspection Script
=========================
This script verifies and visualizes the MNIST training dataset loaded 
through src/data_loader.py.

What it does:
1. Loads the MNIST training dataset using get_mnist_dataloader.
2. Prints dataset summary (sample count, batch shapes, dtype, pixel min/max, label range).
3. Extracts 25 sample images and their corresponding labels.
4. Denormalizes images from [-1, 1] back to [0, 1] for visual display.
5. Saves a clean 5x5 preview grid to results/real_samples_grid.png.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Ensure project root is in Python's search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import get_mnist_dataloader


def inspect_dataset():
    print("=" * 60)
    print("      PHASE 1: MNIST DATASET INSPECTION & VERIFICATION      ")
    print("=" * 60)

    # 1. Load the data using our pipeline
    batch_size = 64
    dataloader, dataset = get_mnist_dataloader(batch_size=batch_size, dataset_dir="./dataset")

    # 2. Check total dataset size
    total_images = len(dataset)
    print(f"\n[1] Total Training Images: {total_images}")

    # 3. Retrieve a single batch to inspect its structure
    images, labels = next(iter(dataloader))

    print(f"\n[2] Batch Shapes:")
    print(f"    - Images batch shape: {images.shape}  [batch_size, channels, height, width]")
    print(f"    - Labels batch shape: {labels.shape}  [batch_size]")

    # 4. Check data types and pixel value range
    print(f"\n[3] Tensor Details:")
    print(f"    - Image Tensor Data Type: {images.dtype}")
    print(f"    - Label Tensor Data Type: {labels.dtype}")
    print(f"    - Minimum pixel value in batch: {images.min().item():.4f} (expected ~ -1.0)")
    print(f"    - Maximum pixel value in batch: {images.max().item():.4f} (expected ~ +1.0)")

    # 5. Check labels
    print(f"\n[4] Class Label Information:")
    print(f"    - Unique labels in this batch: {sorted(labels.unique().tolist())}")
    print(f"    - Minimum label: {labels.min().item()} | Maximum label: {labels.max().item()} (digits 0 to 9)")

    # 6. Create a 5x5 grid (25 images) to display and save
    os.makedirs("./results", exist_ok=True)
    output_path = "./results/real_samples_grid.png"

    print(f"\n[5] Generating a 5x5 sample grid (25 real images)...")
    fig, axes = plt.subplots(5, 5, figsize=(7, 7))
    fig.suptitle("Real MNIST Handwritten Digits (Condition: Labels 0-9)", fontsize=13, fontweight="bold")

    for idx in range(25):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]

        # Extract single image tensor with shape (1, 28, 28)
        img_tensor = images[idx]
        lbl = labels[idx].item()

        # Denormalize: map [-1, 1] back to [0, 1] for correct visual rendering
        # Formula: display_image = (normalized_image * 0.5) + 0.5
        img_denorm = (img_tensor * 0.5) + 0.5
        img_denorm = torch.clamp(img_denorm, 0.0, 1.0)

        # Squeeze channel dimension (1, 28, 28) -> (28, 28) for matplotlib
        img_numpy = img_denorm.squeeze(0).numpy()

        ax.imshow(img_numpy, cmap="gray")
        ax.set_title(f"Label: {lbl}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"    -> Sample grid successfully saved to: {output_path}")
    print("=" * 60)
    print("Dataset verification passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()
