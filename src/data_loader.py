"""
Data Loader Module for MNIST Dataset
====================================
This file handles downloading the MNIST dataset and preparing it for 
training our Conditional Generative Adversarial Network (cGAN).

What it does:
1. Downloads MNIST handwritten digits if not already present.
2. Converts raw 28x28 grayscale images into PyTorch tensors.
3. Normalizes pixel values from [0, 1] to [-1, 1] to match the Generator's Tanh activation.
4. Packages images and labels into mini-batches and shuffles them for training.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(batch_size=64, dataset_dir="./dataset"):
    """
    Creates and returns a PyTorch DataLoader for the MNIST training set.

    Parameters:
        batch_size (int): Number of images per training batch (default: 64).
        dataset_dir (str): Folder path where MNIST will be saved (default: './dataset').

    Returns:
        dataloader (DataLoader): PyTorch DataLoader yielding batches of (images, labels).
        dataset (Dataset): The underlying PyTorch MNIST dataset object.
    """
    # Ensure the dataset directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    # Define the image transformations
    # Step 1: ToTensor() converts PIL image (values 0-255) to a Tensor (values 0.0 to 1.0)
    #         and changes dimensions from (H, W) to (C, H, W) -> (1, 28, 28)
    # Step 2: Normalize with mean=0.5, std=0.5 maps [0.0, 1.0] to [-1.0, 1.0]
    #         Formula: normalized_pixel = (pixel - 0.5) / 0.5
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    # Download and load the training split of MNIST
    dataset = datasets.MNIST(
        root=dataset_dir,
        train=True,            # We use the 60,000 training images
        download=True,         # Auto-download from official mirror if not found
        transform=transform    # Apply our tensor conversion and normalization
    )

    # Wrap the dataset in a DataLoader
    # - batch_size: loads 64 images at a time instead of all 60,000 at once
    # - shuffle=True: randomly scrambles the order every epoch so the model learns general features
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader, dataset


if __name__ == "__main__":
    # Quick self-test when running this file directly
    print("Testing data loader module...")
    loader, data = get_mnist_dataloader(batch_size=64)
    print(f"Total training samples: {len(data)}")
    first_images, first_labels = next(iter(loader))
    print(f"First batch images shape: {first_images.shape}")
    print(f"First batch labels shape: {first_labels.shape}")
    print("Data loader is ready!")
