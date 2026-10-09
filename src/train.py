"""
Training Script for Conditional GAN (cGAN)
==========================================
This script trains the Generator and Discriminator models on MNIST digits.

Key components:
1. Two networks: Generator (forger) and Discriminator (critic).
2. Loss function: BCEWithLogitsLoss (numerically stable binary cross-entropy with logits).
3. Optimizers: Adam with standard GAN learning rate (0.0002) and beta1 (0.5).
4. Fixed noise vector to monitor visual generation progress across epochs.
5. Saves model checkpoints and preview images locally.
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Ensure project root is in the Python search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import get_mnist_dataloader
from src.model import Generator, Discriminator


def save_preview_grid(generator, fixed_noise, fixed_labels, epoch, output_dir="./results"):
    """
    Generates and saves a preview grid of digit images using a fixed noise vector.
    This lets us visually track model quality improvements epoch by epoch.
    """
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()

    with torch.no_grad():
        fake_images = generator(fixed_noise, fixed_labels)
        # Denormalize from [-1, 1] back to [0, 1] for saving
        fake_images = (fake_images * 0.5) + 0.5
        fake_images = torch.clamp(fake_images, 0.0, 1.0)

        # Make a grid where each row contains samples of the same digit
        # 10 rows (digits 0 to 9), 5 columns per digit = 50 images
        grid_path = os.path.join(output_dir, f"preview_epoch_{epoch}.png")
        vutils.save_image(fake_images, grid_path, nrow=5, normalize=False)

        # Also save as latest_preview.png for quick viewing
        latest_path = os.path.join(output_dir, "latest_preview.png")
        vutils.save_image(fake_images, latest_path, nrow=5, normalize=False)

    generator.train()
    print(f"    -> Preview image saved to: {grid_path}")


def train_cgan(epochs=10, batch_size=64, lr=0.0002, latent_dim=100, smoke_test=False, max_batches=None):
    """
    Main training function for the Conditional GAN.

    Parameters:
        epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        lr (float): Learning rate for Adam optimizer.
        latent_dim (int): Length of the random noise vector z.
        smoke_test (bool): If True, runs only a few batches for fast verification.
        max_batches (int or None): Max batches to run per epoch (for smoke testing).
    """
    print("=" * 65)
    print("             PHASE 3: CONDITIONAL GAN TRAINING PIPELINE            ")
    print("=" * 65)

    # 1. Device selection: CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] Using compute device: {device}")
    if smoke_test:
        print("[!] RUNNING IN SMOKE-TEST MODE (small verification run)")

    # 2. Directories for checkpoints and results
    checkpoint_dir = "./checkpoints"
    results_dir = "./results"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 3. Load MNIST training data
    print("\n[2] Loading MNIST dataset...")
    dataloader, dataset = get_mnist_dataloader(batch_size=batch_size, dataset_dir="./dataset")
    print(f"    - Total dataset size: {len(dataset)} images")
    print(f"    - Total batches per epoch: {len(dataloader)} (batch size: {batch_size})")

    # 4. Instantiate Generator and Discriminator
    num_classes = 10
    generator = Generator(latent_dim=latent_dim, num_classes=num_classes).to(device)
    discriminator = Discriminator(num_classes=num_classes).to(device)

    # 5. Loss Function: Binary Cross-Entropy with Logits
    # Numerically stable combination of Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss()

    # 6. Optimizers: Adam with standard GAN hyperparameter beta1=0.5
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # 7. Create a fixed noise vector for repeatable preview grids
    # We create 5 samples for each of the 10 digits (50 total images)
    # Digits: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ..., 9, 9, 9, 9, 9]
    fixed_labels = torch.tensor([digit for digit in range(10) for _ in range(5)], dtype=torch.int64, device=device)
    fixed_noise = torch.randn(len(fixed_labels), latent_dim, device=device)

    # Metric tracking
    epoch_g_losses = []
    epoch_d_losses = []

    print("\n[3] Starting training loop...")

    for epoch in range(1, epochs + 1):
        running_g_loss = 0.0
        running_d_loss = 0.0
        batch_count = 0

        for batch_idx, (real_images, real_labels) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            current_batch_size = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)

            # Target labels for loss computation:
            # 1.0 = Real, 0.0 = Fake
            real_targets = torch.ones(current_batch_size, 1, device=device)
            fake_targets = torch.zeros(current_batch_size, 1, device=device)

            # =========================================================
            # Step A: Train Discriminator (Maximize log(D(x)) + log(1 - D(G(z))))
            # =========================================================
            optimizer_d.zero_grad()

            # 1. Train on real MNIST images
            d_real_logits = discriminator(real_images, real_labels)
            d_loss_real = criterion(d_real_logits, real_targets)

            # 2. Generate fake images from random noise and random class labels
            noise = torch.randn(current_batch_size, latent_dim, device=device)
            random_fake_labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
            fake_images = generator(noise, random_fake_labels)

            # 3. Train on fake images
            # .detach() stops gradients from propagating back to the Generator during Discriminator update
            d_fake_logits = discriminator(fake_images.detach(), random_fake_labels)
            d_loss_fake = criterion(d_fake_logits, fake_targets)

            # Total Discriminator Loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # =========================================================
            # Step B: Train Generator (Maximize log(D(G(z))))
            # =========================================================
            optimizer_g.zero_grad()

            # The Generator wants the Discriminator to classify its fake images as REAL (target = 1.0)
            # Notice we pass the attached fake_images so gradients flow into the Generator weights
            d_fake_for_generator = discriminator(fake_images, random_fake_labels)
            g_loss = criterion(d_fake_for_generator, real_targets)

            g_loss.backward()
            optimizer_g.step()

            # Accumulate losses
            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
            batch_count += 1

            # Print batch progress in regular mode every 200 batches
            if not smoke_test and (batch_idx + 1) % 200 == 0:
                print(f"    Epoch [{epoch}/{epochs}] | Batch [{batch_idx + 1}/{len(dataloader)}] "
                      f"| D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # Compute average losses for this epoch
        avg_g_loss = running_g_loss / batch_count
        avg_d_loss = running_d_loss / batch_count
        epoch_g_losses.append(avg_g_loss)
        epoch_d_losses.append(avg_d_loss)

        print(f"\n Epoch [{epoch}/{epochs}] Complete -> "
              f"Avg D Loss: {avg_d_loss:.4f} | Avg G Loss: {avg_g_loss:.4f}")

        # Save preview images
        save_preview_grid(generator, fixed_noise, fixed_labels, epoch, output_dir=results_dir)

        # Save checkpoint after each epoch
        g_checkpoint_path = os.path.join(checkpoint_dir, "generator_checkpoint.pth")
        d_checkpoint_path = os.path.join(checkpoint_dir, "discriminator_checkpoint.pth")

        torch.save({
            "epoch": epoch,
            "latent_dim": latent_dim,
            "num_classes": num_classes,
            "model_state_dict": generator.state_dict(),
            "optimizer_state_dict": optimizer_g.state_dict(),
            "loss": avg_g_loss
        }, g_checkpoint_path)

        torch.save({
            "epoch": epoch,
            "num_classes": num_classes,
            "model_state_dict": discriminator.state_dict(),
            "optimizer_state_dict": optimizer_d.state_dict(),
            "loss": avg_d_loss
        }, d_checkpoint_path)

        print(f"    -> Checkpoints saved to: {checkpoint_dir}/")

    # Save recorded loss history to a JSON file for plotting in Phase 5
    loss_history_path = os.path.join(results_dir, "loss_history.json")
    with open(loss_history_path, "w") as f:
        json.dump({
            "epoch_g_losses": epoch_g_losses,
            "epoch_d_losses": epoch_d_losses
        }, f, indent=4)
    print(f"    -> Loss history saved to: {loss_history_path}")

    print("\n" + "=" * 65)
    print("Training procedure completed successfully!")
    print("=" * 65)

    return {
        "generator": generator,
        "discriminator": discriminator,
        "g_losses": epoch_g_losses,
        "d_losses": epoch_d_losses,
        "g_checkpoint": g_checkpoint_path,
        "d_checkpoint": d_checkpoint_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditional GAN on MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam optimizer")
    parser.add_argument("--latent-dim", type=int, default=100, help="Dimension of latent noise vector z")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test (few batches)")
    parser.add_argument("--max-batches", type=int, default=None, help="Maximum batches per epoch")

    args = parser.parse_args()

    if args.smoke_test and args.max_batches is None:
        args.max_batches = 5  # Run 5 batches during smoke test

    train_cgan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        smoke_test=args.smoke_test,
        max_batches=args.max_batches
    )
