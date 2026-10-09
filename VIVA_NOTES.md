# Conditional GAN (cGAN) Project — Comprehensive Viva Study Guide

This document is specifically tailored to help you answer questions confidently in your college lab viva, external project review, and theoretical exams.

---

## Part 1: Core Theoretical Concepts in Simple Language

### 1. What is a Generative Adversarial Network (GAN)?
* **Analogy:** Think of an **art forger** (Generator) and an **art detective/inspector** (Discriminator).
* The forger has never seen the original paintings; they only receive feedback on whether their forgeries pass as genuine.
* The inspector looks at both genuine masterpieces (real MNIST data) and forgeries, learning to spot telltale flaws.
* Over time, both players improve simultaneously through competitive game theory (minimax optimization).

### 2. What is a Conditional GAN (cGAN)?
* In a standard, **unconditional GAN**, the generator takes purely random noise and produces an arbitrary image from the dataset (e.g., any random digit from 0 to 9). You cannot choose *which* digit it creates.
* In a **Conditional GAN (cGAN)**, we provide an additional condition (a class label, such as "draw a 7") to **both** the Generator and the Discriminator.
* This allows **directed, class-specific synthesis** on demand.

### 3. The Generator ($G$)
* **Role:** Maps a low-dimensional latent noise vector $z$ plus a class label $y$ into a high-dimensional image tensor $\hat{x}$.
* **Our Architecture:**
  - Input: $100$ noise values $+ 10$ one-hot class values $= 110$.
  - Hidden Layers: `Linear(110 -> 256)` with `ReLU`, followed by `Linear(256 -> 512)` with `ReLU`.
  - Output Layer: `Linear(512 -> 784)` with `Tanh` activation.
  - Final Shape: Reshaped from $784$ to $[1, 28, 28]$ (grayscale image).
* **Why Tanh?** `Tanh` bounds all pixel values strictly between $[-1.0, 1.0]$, exactly matching our dataset normalization.

### 4. The Discriminator ($D$)
* **Role:** A binary classifier that evaluates whether an $(image, label)$ pair is real or synthetic.
* **Our Architecture:**
  - Input: $784$ flattened image pixels $+ 10$ one-hot class values $= 794$.
  - Hidden Layers: `Linear(794 -> 512)` with `LeakyReLU(0.2)`, followed by `Linear(512 -> 256)` with `LeakyReLU(0.2)`.
  - Output Layer: `Linear(256 -> 1)` producing a raw logit.
* **Why LeakyReLU?** Standard `ReLU` zeroes out negative activations, creating dead neurons where gradients cannot flow. `LeakyReLU` allows a small gradient ($0.2 \times x$) for negative values, keeping learning alive.
* **Why No Sigmoid in the Final Layer?** We use PyTorch's `nn.BCEWithLogitsLoss()`. It applies the sigmoid activation internally using log-sum-exp arithmetic, which is numerically stable and avoids floating-point underflow.

### 5. Latent Space & Noise Vector ($z$)
* The latent vector $z$ is drawn from a standard normal distribution $\mathcal{N}(0, 1)$.
* It provides the "creative randomness" or diversity for the model. Without noise, the network would deterministically output the exact same identical image for each label.
* Variations in $z$ correspond to differences in writing style, stroke width, slant, and positioning.

### 6. One-Hot Class Encoding
* Class labels ($0$ to $9$) are integers. Passing a single integer directly could mislead the linear layers into treating class $9$ as numerically "nine times greater" than class $1$.
* One-hot encoding converts each class into a 10-dimensional binary vector where only the target class index is $1$ and all other indices are $0$:
  - Label $0 \rightarrow [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]$
  - Label $7 \rightarrow [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]$

### 7. Adversarial Loss & Targets
* **Real Targets = 1.0**: Assigned to authentic MNIST training images.
* **Fake Targets = 0.0**: Assigned to Generator-created images.
* **Discriminator Objective:** Maximize $\log D(x, y) + \log(1 - D(G(z, y), y))$.
* **Generator Objective:** Maximize $\log D(G(z, y), y)$, which is implemented by minimizing binary cross-entropy against target $1.0$.

### 8. Why do we call `fake_images.detach()`?
* When updating the Discriminator, we feed it generated images.
* If we did not detach the tensor, PyTorch would backpropagate gradients all the way through the Generator network as well.
* Calling `.detach()` cuts the computational graph at the fake image tensor, ensuring **only** Discriminator weights are updated during the Discriminator's training step.

---

## Part 2: Project-Specific Observations & Honest Findings

### 1. Discriminator Dominance (Our Recorded 10-Epoch Result)
* In our training run, Discriminator loss decreased from $0.6084 \rightarrow 0.0003$, while Generator loss rose from $2.83 \rightarrow 12.01$.
* **Viva Explanation:** This is a classic, documented phenomenon called **Discriminator Dominance**. The Discriminator was able to spot artificial artifacts faster than the simple MLP Generator could learn to correct them. When the Discriminator is near-perfect, its gradients diminish, slowing the Generator's progress.

### 2. MLP vs. Convolutional Architectures
* Our project uses fully connected linear layers (MLP) because they are simple, transparent, and train quickly on CPU (~30s/epoch).
* However, MLPs flatten images into 1D vectors of 784 numbers, losing 2D spatial context.
* A natural enhancement is a Deep Convolutional GAN (DCGAN), which uses 2D convolution kernels to maintain spatial relationships and generate sharper edges.

---

## Part 3: Top 15 Viva Questions and Answers

### Q1: What is the main objective of this project?
**Answer:** To implement a class-conditional Generative Adversarial Network (cGAN) using PyTorch that generates handwritten digits ($0$ to $9$) from the MNIST dataset conditioned on user-specified class labels.

### Q2: What is the difference between a standard GAN and a Conditional GAN (cGAN)?
**Answer:** A standard GAN generates random images from the dataset with no user control over the output class. A Conditional GAN conditions both the Generator and Discriminator on auxiliary information (such as a class label), enabling targeted generation of specific classes.

### Q3: Why is the class label fed to both the Generator and the Discriminator?
**Answer:** The Generator needs the label to know which digit to draw. The Discriminator needs the label to verify not only that the image looks like a real digit, but that it matches the claimed class (e.g., rejecting an image of an "8" if the requested condition was "3").

### Q4: How is image normalization configured in this project?
**Answer:** Real MNIST images are normalized to $[-1, 1]$ using `transforms.Normalize((0.5,), (0.5,))`. This directly matches the output range of the Generator's final `Tanh` activation function.

### Q5: What is the role of latent noise $z$?
**Answer:** Latent noise is a 100-dimensional vector sampled from a standard normal distribution $\mathcal{N}(0, 1)$. It provides stochastic diversity so that multiple generations of the same digit exhibit varying handwriting styles, slants, and stroke thicknesses.

### Q6: Why do we use one-hot encoding for the class condition?
**Answer:** One-hot encoding creates orthogonal 10-dimensional vectors, ensuring each digit class is treated as an independent categorical identity without artificial numerical hierarchy.

### Q7: Why does the Discriminator use `LeakyReLU` instead of standard `ReLU`?
**Answer:** Standard `ReLU` zeroes out negative inputs, which can permanently deactivate neurons ("dying ReLU"). `LeakyReLU(0.2)` maintains a small slope for negative values, ensuring gradients flow even when the Discriminator is confident.

### Q8: Why does the Discriminator not have a Sigmoid layer in `src/model.py`?
**Answer:** The Discriminator outputs raw logits because we use `nn.BCEWithLogitsLoss()`. This loss combines the sigmoid function and cross-entropy loss into a single mathematical step that uses log-sum-exp stabilization to prevent numerical underflow.

### Q9: What does `fake_images.detach()` do during Discriminator training?
**Answer:** It prevents gradients from flowing back into the Generator's parameters while updating the Discriminator, isolating the Discriminator's optimization step.

### Q10: How are the Generator and Discriminator optimized?
**Answer:** They are updated alternately using two separate Adam optimizers with a learning rate of $0.0002$ and momentum $\beta_1 = 0.5$. Lowering $\beta_1$ from standard $0.9$ to $0.5$ dampens oscillations in adversarial training.

### Q11: In standard classifiers, a loss approaching zero is ideal. Why is $D_{\text{loss}} \approx 0$ undesirable in GANs?**
**Answer:** When $D_{\text{loss}} \rightarrow 0$, the Discriminator is effortlessly beating the Generator. Its output saturates, causing vanishing gradients that starve the Generator of the feedback needed to improve.

### Q12: What does the experiment in `condition_comparison.png` prove?
**Answer:** It passes the exact same latent noise vector across all 10 digit classes. Holding noise constant while varying only the class condition proves that the class label—and not random noise—is what drives digit identity.

### Q13: What is mode collapse?
**Answer:** Mode collapse is a failure mode where the Generator discovers a small subset of outputs (e.g., only generating digit "1") that fools the Discriminator, and stops generating diverse variations or other classes entirely.

### Q14: What were the key limitations observed in our 10-epoch training run?
**Answer:** 
1. The simple MLP lacks spatial convolutional inductive bias, producing some edge speckle noise.
2. The Discriminator overpowered the Generator early on ($D_{\text{loss}} \rightarrow 0.0003$), slowing Generator refinement.

### Q15: How can this model be improved in future work?
**Answer:** 
1. Transition to a Deep Convolutional GAN (DCGAN) using 2D convolutions.
2. Apply one-sided label smoothing (target $0.9$ instead of $1.0$) or dropout to prevent Discriminator dominance.
3. Use learned `nn.Embedding` vectors for richer condition representations.
