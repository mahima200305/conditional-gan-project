# Class-Conditional Image Generation using a Conditional GAN (cGAN)

A beginner-friendly PyTorch implementation of a **Conditional Generative Adversarial Network (cGAN)** trained on the MNIST handwritten digits dataset.

---

## 📌 Project Overview
The objective of this project is to build a machine learning model capable of generating specific handwritten digits (0 through 9) on command based on an input label (condition).

- **Input Condition:** Class label $y \in \{0, 1, ..., 9\}$ along with a random noise vector $z$.
- **Generated Output:** A $28 \times 28$ grayscale handwritten digit image corresponding to the label.

---

## 📂 Project Structure
```text
conditional-gan-project/
├── .gitignore             # Files and folders ignored by Git
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── dataset/               # Auto-downloaded MNIST dataset (ignored by Git)
├── src/                   # Source code
│   ├── model.py           # Generator and Discriminator neural networks (Phase 2)
│   ├── train.py           # Model training script (Phase 3)
│   └── generate.py        # Image generation script (Phase 4)
├── notebooks/             # Jupyter notebooks for interactive analysis (Phase 5)
│   └── cgan_analysis.ipynb
├── results/               # Generated sample images and evaluation plots
└── screenshots/           # Execution screenshots and demonstration images
```

---

## 🛠️ Technology Stack
- **Language:** Python
- **Deep Learning Framework:** PyTorch & Torchvision
- **Numerical & Plotting:** NumPy, Matplotlib
- **Dataset:** MNIST (Handwritten Digits)
- **Environment:** Windows, VS Code, Git

---

## 🚀 Quick Setup

1. **Activate the Virtual Environment:**
   - PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Command Prompt:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```

2. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## 📋 Project Roadmap
- [x] **Phase 0:** Project Setup & Environment Configuration
- [ ] **Phase 1:** Dataset Preparation (MNIST pipeline & exploration)
- [ ] **Phase 2:** Build the cGAN Models (Generator & Discriminator)
- [ ] **Phase 3:** Train the cGAN
- [ ] **Phase 4:** Generate Images for Different Conditions
- [ ] **Phase 5:** Analyze and Visualize Results
- [ ] **Phase 6:** Final Documentation & Report
