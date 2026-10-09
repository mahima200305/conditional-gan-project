"""
Class-Conditional Image Generation Script
=========================================
This script loads the trained Generator from its saved checkpoint and 
generates handwritten digits conditioned on requested class labels (0-9).

Key capabilities:
1. Generates multiple variations of a user-selected digit (0-9).
2. Generates a condition comparison grid showing all 10 digits generated 
   using the exact same latent noise vectors, visually proving class conditioning.
3. Saves high-resolution results to results/generated_digits.png and 
   results/condition_comparison.png.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# Ensure project root is in the Python search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.model import Generator


def load_generator(checkpoint_path="checkpoints/generator_checkpoint.pth", device="cpu"):
    """
    Safely loads the Generator model and its trained weights from a checkpoint file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            f"Please run 'src/train.py' first to train and save the model."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    latent_dim = checkpoint.get("latent_dim", 100)
    num_classes = checkpoint.get("num_classes", 10)
    epoch = checkpoint.get("epoch", "unknown")

    print(f"[+] Loaded Generator checkpoint from epoch: {epoch}")
    print(f"    - Latent dimension: {latent_dim}")
    print(f"    - Number of classes: {num_classes}")

    generator = Generator(latent_dim=latent_dim, num_classes=num_classes)
    generator.load_state_dict(checkpoint["model_state_dict"])
    generator.to(device)
    generator.eval()  # Set to evaluation mode for inference

    return generator, latent_dim, num_classes, epoch


def generate_single_class_samples(generator, digit, num_samples=16, latent_dim=100, device="cpu", output_path="results/generated_digits.png"):
    """
    Generates multiple diverse examples for one specific digit label.
    """
    if not (0 <= digit <= 9):
        raise ValueError(f"Requested digit must be between 0 and 9, got: {digit}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Generate random noise vectors for diversity
    noise = torch.randn(num_samples, latent_dim, device=device)

    # 2. Condition all samples on the requested digit
    labels = torch.full((num_samples,), fill_value=digit, dtype=torch.int64, device=device)

    # 3. Generate images with no gradient tracking
    with torch.no_grad():
        generated = generator(noise, labels)

    # 4. Denormalize from [-1, 1] to [0, 1] for visual display
    images = (generated * 0.5) + 0.5
    images = torch.clamp(images, 0.0, 1.0).cpu()

    # 5. Plot in a 4x4 grid (or suitable rectangle)
    rows = int(num_samples ** 0.5)
    cols = (num_samples + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(f"Generated Samples Conditioned on Digit: '{digit}'", fontsize=14, fontweight="bold")

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i in range(len(axes_flat)):
        if i < num_samples:
            img = images[i].squeeze(0).numpy()
            axes_flat[i].imshow(img, cmap="gray")
            axes_flat[i].set_title(f"Sample {i + 1}", fontsize=9)
        axes_flat[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] Saved {num_samples} samples of digit '{digit}' to: {output_path}")

    return images


def generate_condition_comparison(generator, num_styles=5, latent_dim=100, device="cpu", output_path="results/condition_comparison.png"):
    """
    Generates a comparison grid across all 10 digits (0 to 9).
    Crucially, each row uses the SAME latent noise vector across all 10 columns.
    This demonstrates that changing only the class condition produces the corresponding digit.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create fixed noise vectors: 1 noise vector per style row
    # Shape: [num_styles, latent_dim]
    base_noise = torch.randn(num_styles, latent_dim, device=device)

    fig, axes = plt.subplots(num_styles, 10, figsize=(15, num_styles * 1.6))
    fig.suptitle("Condition Comparison Grid (Columns = Class 0-9 | Rows = Same Noise Vector)", fontsize=13, fontweight="bold")

    with torch.no_grad():
        for style_idx in range(num_styles):
            # Take one specific noise vector and repeat it for all 10 classes
            noise_row = base_noise[style_idx:style_idx + 1].repeat(10, 1)  # Shape: [10, latent_dim]
            class_labels = torch.arange(0, 10, dtype=torch.int64, device=device)  # [0, 1, ..., 9]

            # Generate digits 0-9 with the identical noise
            row_images = generator(noise_row, class_labels)
            row_images = (row_images * 0.5) + 0.5
            row_images = torch.clamp(row_images, 0.0, 1.0).cpu()

            for class_idx in range(10):
                ax = axes[style_idx, class_idx] if num_styles > 1 else axes[class_idx]
                img = row_images[class_idx].squeeze(0).numpy()
                ax.imshow(img, cmap="gray")
                if style_idx == 0:
                    ax.set_title(f"Class {class_idx}", fontsize=11, fontweight="bold")
                if class_idx == 0:
                    ax.set_ylabel(f"Style {style_idx + 1}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] Saved condition comparison grid across classes 0-9 to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate MNIST images conditioned on class labels")
    parser.add_argument("--digit", type=int, default=7, help="Digit to generate (0 to 9, default: 7)")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of examples of chosen digit")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/generator_checkpoint.pth", help="Path to generator checkpoint")
    parser.add_argument("--styles", type=int, default=5, help="Number of noise styles for comparison grid")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load trained generator
    generator, latent_dim, num_classes, epoch = load_generator(args.checkpoint, device=device)

    # 2. Generate samples of the selected digit
    digits_output = "results/generated_digits.png"
    generate_single_class_samples(
        generator=generator,
        digit=args.digit,
        num_samples=args.num_samples,
        latent_dim=latent_dim,
        device=device,
        output_path=digits_output
    )

    # 3. Generate condition comparison across all 10 classes using identical noise
    comparison_output = "results/condition_comparison.png"
    generate_condition_comparison(
        generator=generator,
        num_styles=args.styles,
        latent_dim=latent_dim,
        device=device,
        output_path=comparison_output
    )

    print("\nGeneration finished successfully!")


if __name__ == "__main__":
    main()
