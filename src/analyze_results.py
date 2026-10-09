"""
Results Analysis and Visualization Script
==========================================
This script inspects the actual training losses and generated outputs 
from the Conditional GAN project.

What it does:
1. Loads the actual recorded loss values from results/loss_history.json.
2. Validates that all loss numbers are finite and non-empty.
3. Plots Generator Loss vs. Discriminator Loss across all training epochs.
4. Saves the plot to results/training_loss_plot.png.
5. Prints honest statistical metrics without inventing unmeasured benchmarks.
"""

import os
import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_training_losses(json_path="results/loss_history.json", output_plot_path="results/training_loss_plot.png"):
    """
    Loads recorded loss history, performs numerical sanity checks, 
    and generates a dual-curve loss plot.
    """
    print("=" * 65)
    print("          PHASE 5: cGAN TRAINING LOSS ANALYSIS & VISUALIZATION      ")
    print("=" * 65)

    # 1. Check if file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Loss history file not found at: '{json_path}'. "
            "Please ensure 'src/train.py' was executed."
        )

    # 2. Load JSON data
    with open(json_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted or invalid JSON format in '{json_path}': {e}")

    # 3. Validate keys and contents
    if "epoch_g_losses" not in data or "epoch_d_losses" not in data:
        raise KeyError(
            f"JSON must contain 'epoch_g_losses' and 'epoch_d_losses'. Found keys: {list(data.keys())}"
        )

    g_losses = data["epoch_g_losses"]
    d_losses = data["epoch_d_losses"]

    if len(g_losses) == 0 or len(d_losses) == 0:
        raise ValueError("Recorded loss lists are empty. No data to analyze.")

    if len(g_losses) != len(d_losses):
        raise ValueError(
            f"Length mismatch: g_losses has {len(g_losses)} epochs, d_losses has {len(d_losses)} epochs."
        )

    num_epochs = len(g_losses)
    epochs = list(range(1, num_epochs + 1))

    # 4. Check that all values are finite numbers
    for epoch_idx, (g_l, d_l) in enumerate(zip(g_losses, d_losses), start=1):
        if not np.isfinite(g_l):
            raise ValueError(f"Non-finite Generator loss detected at epoch {epoch_idx}: {g_l}")
        if not np.isfinite(d_l):
            raise ValueError(f"Non-finite Discriminator loss detected at epoch {epoch_idx}: {d_l}")

    print(f"\n[1] Data Validation Summary:")
    print(f"    - Total Epochs Recorded: {num_epochs}")
    print(f"    - Data Format: Epoch-averaged BCEWithLogitsLoss values")
    print(f"    - Finite Value Check: PASSED (all values finite floats)")

    # 5. Compute summary statistics
    print(f"\n[2] Observed Loss Statistics:")
    print(f"    - Generator Loss:")
    print(f"        * Initial (Epoch 1): {g_losses[0]:.4f}")
    print(f"        * Final   (Epoch {num_epochs}): {g_losses[-1]:.4f}")
    print(f"        * Min: {min(g_losses):.4f} | Max: {max(g_losses):.4f} | Mean: {np.mean(g_losses):.4f}")
    print(f"    - Discriminator Loss:")
    print(f"        * Initial (Epoch 1): {d_losses[0]:.4f}")
    print(f"        * Final   (Epoch {num_epochs}): {d_losses[-1]:.4f}")
    print(f"        * Min: {min(d_losses):.4f} | Max: {max(d_losses):.4f} | Mean: {np.mean(d_losses):.4f}")

    # 6. Plotting
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.figure(figsize=(9, 5.5))

    plt.plot(epochs, g_losses, label="Generator Loss (G)", color="#1f77b4", linewidth=2.2, marker="o", markersize=5)
    plt.plot(epochs, d_losses, label="Discriminator Loss (D)", color="#d62728", linewidth=2.2, marker="s", markersize=5)

    plt.title("Conditional GAN Training Loss Curves (10 Epochs on MNIST)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Training Epoch", fontsize=11)
    plt.ylabel("BCEWithLogitsLoss", fontsize=11)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="center right")
    plt.tight_layout()

    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    print(f"\n[3] Saved training loss plot to: {output_plot_path}")

    print("=" * 65)
    print("Loss analysis completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    analyze_training_losses()
