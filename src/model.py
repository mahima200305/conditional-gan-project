"""
Conditional GAN (cGAN) Models Module
====================================
This file defines the two neural networks used in our Conditional GAN:
1. Generator: Learns to generate synthetic 28x28 digit images conditioned on a digit label.
2. Discriminator: Evaluates if a given 28x28 image and its label pairing are real or fake.

Both networks use one-hot encoding for the class labels to easily condition the generation process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generator Network for Conditional GAN.

    It takes:
      1. A random noise vector 'z' (latent representation, size: latent_dim).
      2. A target class label (integer from 0 to 9).

    It converts the label into a one-hot vector, concatenates it with the noise,
    and passes the combined vector through fully connected layers to produce
    a 28x28 grayscale image.
    """
    def __init__(self, latent_dim=100, num_classes=10, image_channels=1, image_size=28):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_channels = image_channels
        self.image_size = image_size
        self.output_pixels = image_channels * image_size * image_size  # 1 * 28 * 28 = 784

        # Total input size = noise dimension + one-hot condition dimension (e.g., 100 + 10 = 110)
        input_dim = self.latent_dim + self.num_classes

        # Fully connected layers to synthesize the image
        self.network = nn.Sequential(
            # Hidden layer 1: 110 -> 256
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),

            # Hidden layer 2: 256 -> 512
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),

            # Output layer: 512 -> 784 pixels
            nn.Linear(512, self.output_pixels),

            # Tanh activation squashes all pixel values into [-1, 1],
            # which matches the normalization applied to our real MNIST images.
            nn.Tanh()
        )

    def forward(self, noise, labels):
        """
        Forward pass for the Generator.

        Parameters:
            noise (Tensor): Batch of random latent vectors of shape [batch_size, latent_dim].
            labels (Tensor): Batch of integer class labels (0-9) of shape [batch_size].

        Returns:
            images (Tensor): Generated image tensor of shape [batch_size, 1, 28, 28].
        """
        batch_size = noise.size(0)

        # 1. Convert integer labels (e.g., 3) into 10-dimensional one-hot vectors
        # Shape: [batch_size, 10]
        label_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

        # 2. Concatenate the noise vector and the one-hot label vector along dimension 1
        # Shape: [batch_size, latent_dim + 10] = [batch_size, 110]
        generator_input = torch.cat([noise, label_one_hot], dim=1)

        # 3. Pass through the fully connected network
        # Shape: [batch_size, 784]
        flattened_image = self.network(generator_input)

        # 4. Reshape the 784-element 1D vector into a 2D image: [batch_size, 1, 28, 28]
        generated_images = flattened_image.view(batch_size, self.image_channels, self.image_size, self.image_size)

        return generated_images


class Discriminator(nn.Module):
    """
    Discriminator Network for Conditional GAN.

    It takes:
      1. An image tensor of shape [batch_size, 1, 28, 28].
      2. The corresponding class label of shape [batch_size].

    It flattens the image into 784 pixels, converts the label into a one-hot vector,
    concatenates them together, and produces a single scalar logit per image
    indicating whether the (image, label) pair is real or fake.
    """
    def __init__(self, num_classes=10, image_channels=1, image_size=28):
        super(Discriminator, self).__init__()

        self.num_classes = num_classes
        self.image_channels = image_channels
        self.image_size = image_size
        self.input_pixels = image_channels * image_size * image_size  # 1 * 28 * 28 = 784

        # Total input size = 784 flattened pixels + 10 one-hot values = 794
        input_dim = self.input_pixels + self.num_classes

        # Fully connected layers to classify the pair
        self.network = nn.Sequential(
            # Hidden layer 1: 794 -> 512
            nn.Linear(input_dim, 512),
            # LeakyReLU allows a small positive gradient (0.2) when the unit is inactive,
            # which prevents the "dying neuron" problem during discriminator training.
            nn.LeakyReLU(0.2, inplace=True),

            # Hidden layer 2: 512 -> 256
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer: 256 -> 1 raw real/fake score (logit)
            nn.Linear(256, 1)
            # Notice: No Sigmoid here. We will use BCEWithLogitsLoss during training,
            # which is numerically more stable than applying Sigmoid separately.
        )

    def forward(self, images, labels):
        """
        Forward pass for the Discriminator.

        Parameters:
            images (Tensor): Batch of images of shape [batch_size, 1, 28, 28].
            labels (Tensor): Batch of integer class labels (0-9) of shape [batch_size].

        Returns:
            logits (Tensor): Raw prediction scores of shape [batch_size, 1].
        """
        batch_size = images.size(0)

        # 1. Flatten the 2D image into a 1D vector of 784 pixels: [batch_size, 784]
        flattened_images = images.view(batch_size, -1)

        # 2. Convert integer labels into 10-dimensional one-hot vectors: [batch_size, 10]
        label_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

        # 3. Concatenate flattened image pixels and label one-hot vector: [batch_size, 794]
        discriminator_input = torch.cat([flattened_images, label_one_hot], dim=1)

        # 4. Pass through the network to get raw prediction logits: [batch_size, 1]
        logits = self.network(discriminator_input)

        return logits
