# Phase 5: Experimental Results & Analysis Summary

---

## 1. Experiment Overview & Setup
* **Dataset:** MNIST Handwritten Digits (60,000 training images, normalized to $[-1.0, 1.0]$).
* **Architecture:** Fully Connected Multi-Layer Perceptron (MLP) Conditional GAN:
  * **Generator:** Linear(110 $\rightarrow$ 256) $\rightarrow$ ReLU $\rightarrow$ Linear(256 $\rightarrow$ 512) $\rightarrow$ ReLU $\rightarrow$ Linear(512 $\rightarrow$ 784) $\rightarrow$ Tanh.
  * **Discriminator:** Linear(794 $\rightarrow$ 512) $\rightarrow$ LeakyReLU(0.2) $\rightarrow$ Linear(512 $\rightarrow$ 256) $\rightarrow$ LeakyReLU(0.2) $\rightarrow$ Linear(256 $\rightarrow$ 1).
* **Conditioning Method:** 10-dimensional one-hot encoded vector concatenated with noise (Generator) and flattened image (Discriminator).
* **Optimizer:** Adam ($\text{lr} = 0.0002$, $\beta_1 = 0.5$, $\beta_2 = 0.999$).
* **Loss Function:** `nn.BCEWithLogitsLoss()`.
* **Hardware:** Intel/AMD CPU (Windows local environment).
* **Training Duration:** 10 epochs (938 batches per epoch, batch size = 64).

---

## 2. Actual Measured Loss History
The losses below were recorded as the epoch-averaged `BCEWithLogitsLoss` during the 10-epoch training run in `results/loss_history.json`:

| Epoch | Generator Loss ($G_{\text{loss}}$) | Discriminator Loss ($D_{\text{loss}}$) | Key Dynamics Observed |
| :---: | :---: | :---: | :--- |
| **1** | 2.8312 | 0.6084 | Initial balanced competition; $D$ begins learning digit borders. |
| **2** | 4.9235 | 0.2365 | $D$ becomes substantially more confident in spotting fakes. |
| **3** | 4.5688 | 0.2232 | Temporary stabilization in adversarial gradients. |
| **4** | 5.3330 | 0.1334 | $D_{\text{loss}}$ decreases; $G$ struggles to produce deceptive pixels. |
| **5** | 6.4653 | 0.0511 | $D$ pushes near-zero error on synthetic samples. |
| **6** | 9.0462 | 0.0003 | Discriminator dominance emerges ($D_{\text{loss}} < 0.001$). |
| **7** | 9.8680 | 0.0032 | Vanishing gradient for $G$; loss climbs above 9.0. |
| **8** | 11.0086 | 0.0400 | $D$ easily separates real images from synthetic distributions. |
| **9** | 9.2024 | 0.0042 | Slight fluctuation as $G$ updates parameters. |
| **10** | 12.0189 | 0.0003 | Final state: $D$ maintains near-perfect separation. |

*Visual plot saved at:* [`results/training_loss_plot.png`](file:///c:/conditional-gan-project/results/training_loss_plot.png)

---

## 3. Generated Image Artifacts
The following image files were produced by `src/generate.py` using `checkpoints/generator_checkpoint.pth`:
1. [`results/generated_digits.png`](file:///c:/conditional-gan-project/results/generated_digits.png): 16 samples conditioned on class label `7`.
2. [`results/condition_comparison.png`](file:///c:/conditional-gan-project/results/condition_comparison.png): 5 noise style variations evaluated across all 10 digit conditions ($0$ through $9$).

---

## 4. Qualitative Observations (Visual Inspection)
Direct visual inspection of the generated files reveals the following:
1. **Emergence of Stroke Geometry:** The generator has learned the rough central vertical structure of handwritten strokes against a background.
2. **Background Artifacts & Noise Speckling:** Discrete black and white speckles appear across the border of the $28 \times 28$ bounding box.
3. **Discriminator Dominance Impact:** Because the discriminator loss collapsed to near-zero ($0.0003$), the generator's gradient updates plateaued. Consequently, across the condition comparison grid, the output retains a similar stroke posture rather than achieving distinct, crisp shapes for complex digits like $3, 8, 9$.
4. **Reproducibility:** Across repeated rows in `condition_comparison.png`, the identical noise vector yields consistent structural attributes, verifying that the latent input is correctly wired.

---

## 5. Architectural Limitations & Future Improvements
1. **Lack of Spatial Convolutions (DCGAN):**
   * *Limitation:* The current architecture uses fully connected layers (MLP), flattening the image into 784 numbers. It has no inductive bias for 2D spatial locality.
   * *Improvement:* Transitioning to a Deep Convolutional GAN (DCGAN) using `Conv2d` and `ConvTranspose2d` would greatly improve spatial stroke sharpness.
2. **Discriminator Overpowering:**
   * *Limitation:* The discriminator trained too quickly relative to the generator, causing vanishing gradients for $G$.
   * *Improvement:* Use one-sided label smoothing (e.g., real labels = 0.9 instead of 1.0) or add Batch Normalization (`nn.BatchNorm1d`) and Dropout to regularize the Discriminator.
3. **Conditioning Representation:**
   * *Limitation:* A 10-dimensional one-hot vector concatenated to a 100-dimensional noise vector contributes only ~9% of the input dimension.
   * *Improvement:* Use learned `nn.Embedding` vectors mapped to a higher dimensionality before concatenation, giving the condition stronger influence.

---

*(Note: In accordance with academic honesty, unmeasured quantitative benchmarks such as Frechet Inception Distance (FID), Inception Score (IS), or artificial classification accuracy percentages have not been fabricated.)*
