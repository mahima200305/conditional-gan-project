"""
Model Verification Script
=========================
This script tests the Generator and Discriminator models defined in src/model.py.

It validates:
1. Instantiation of both models.
2. Tensor input and output shapes across both networks.
3. Generator output pixel values are within [-1.0, 1.0] and all values are finite.
4. Discriminator outputs exactly one logit per image sample.
"""

import sys
from pathlib import Path
import torch

# Ensure the project root is in the Python module search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.model import Generator, Discriminator


def run_model_tests():
    print("=" * 60)
    print("        PHASE 2: cGAN MODEL ARCHITECTURE VERIFICATION       ")
    print("=" * 60)

    # 1. Configuration parameters
    batch_size = 4
    latent_dim = 100
    num_classes = 10

    # 2. Instantiate both models
    print("\n[1] Instantiating Generator and Discriminator...")
    generator = Generator(latent_dim=latent_dim, num_classes=num_classes)
    discriminator = Discriminator(num_classes=num_classes)
    print("    -> Models initialized successfully!")

    # 3. Create sample synthetic inputs
    print("\n[2] Creating sample test inputs (Batch size = 4)...")
    noise = torch.randn(batch_size, latent_dim)
    labels = torch.tensor([0, 3, 5, 9], dtype=torch.int64)

    print(f"    - Noise tensor shape:  {list(noise.shape)}   (Expected: [4, 100])")
    print(f"    - Labels tensor shape: {list(labels.shape)}         (Expected: [4])")
    print(f"    - Test condition labels: {labels.tolist()}")

    # 4. Test Generator
    print("\n[3] Testing Generator forward pass...")
    generated_images = generator(noise, labels)
    print(f"    - Generated images shape: {list(generated_images.shape)} (Expected: [4, 1, 28, 28])")

    # Assertions for Generator
    assert generated_images.shape == torch.Size([4, 1, 28, 28]), \
        f"Generator output shape mismatch: {generated_images.shape} != [4, 1, 28, 28]"

    assert torch.all(torch.isfinite(generated_images)), \
        "Generator produced non-finite (NaN or Inf) values!"

    min_pixel = generated_images.min().item()
    max_pixel = generated_images.max().item()
    print(f"    - Generated pixel value range: [{min_pixel:.4f}, {max_pixel:.4f}]")

    assert min_pixel >= -1.0001 and max_pixel <= 1.0001, \
        f"Generated values out of [-1, 1] range: min={min_pixel}, max={max_pixel}"
    print("    -> Generator assertions PASSED (all values finite and bounded in [-1, 1])")

    # 5. Test Discriminator with generated images and conditions
    print("\n[4] Testing Discriminator forward pass...")
    discriminator_logits = discriminator(generated_images, labels)
    print(f"    - Discriminator logits shape: {list(discriminator_logits.shape)} (Expected: [4, 1])")

    # Assertions for Discriminator
    assert discriminator_logits.shape == torch.Size([4, 1]), \
        f"Discriminator output shape mismatch: {discriminator_logits.shape} != [4, 1]"

    assert torch.all(torch.isfinite(discriminator_logits)), \
        "Discriminator produced non-finite (NaN or Inf) values!"
    print("    -> Discriminator assertions PASSED (one real/fake logit per image)")

    # 6. Test Generator sensitivity to different random noises
    noise_2 = torch.randn(batch_size, latent_dim)
    generated_images_2 = generator(noise_2, labels)
    difference = (generated_images - generated_images_2).abs().sum().item()
    assert difference > 0, "Generator produced identical output for different noise vectors!"
    print(f"\n[5] Checked input variation: different random seeds produce distinct initial outputs.")
    print("    (Note: At this stage weights are random, so outputs resemble random noise, not digits yet.)")

    print("\n" + "=" * 60)
    print("ALL MODEL ARCHITECTURE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_model_tests()
