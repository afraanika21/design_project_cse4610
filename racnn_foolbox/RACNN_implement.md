
# Retrieval-Augmented Convolutional Neural Network (RaCNN) — implementation

This repository contains a clean and reproducible PyTorch reimplementation of **Retrieval-Augmented Convolutional Neural Networks (RaCNN)**, originally proposed by Zhao and Cho (2018), for improving adversarial robustness through feature-space retrieval and local mixup.

---

## Dataset

All experiments are conducted on the **CIFAR-10** dataset:
- 50,000 training images and 10,000 test images.
- Images normalized using CIFAR-10 mean and standard deviation:
  - Mean = (0.4914, 0.4822, 0.4465)
  - Std = (0.2023, 0.1994, 0.2010)
- Resolution: 32×32 RGB images.
- The dataset is loaded using `torchvision.datasets.CIFAR10`.

---

## Dependencies

The implementation uses the following main packages:

```bash
torch==2.x
torchvision==0.x
tqdm
numpy
matplotlib
faiss-gpu
foolbox
````

FAISS is used for high-speed similarity retrieval, and Foolbox provides standardized adversarial attack implementations for L2-based robustness testing (FGSM, iFGSM, and DeepFool).

---

## Method Overview

### 1. Baseline Model

A standard 6 layer CNN classifier is trained on normalized CIFAR-10 as the baseline.

### 2. Retrieval-Augmented CNN (RaCNN)

RaCNN enhances robustness by combining supervised learning with feature retrieval and local feature mixing:

* Each test image `x` is passed through a **feature extractor** (ϕ′) to obtain a deep representation.
* Using **FAISS**, the top-K nearest neighbors of `x` are retrieved from the training feature database.
* The features of `x` and its retrieved neighbors are linearly combined using **local mixup**, forming convex combinations that project `x` back toward the data manifold.
* Classification is then performed on this mixed representation through the model’s prediction head.

This mechanism acts as a **feature-space projection** that reduces off-manifold adversarial perturbations.

---

## Adversarial Evaluation

After training, both the baseline and RaCNN models are evaluated under controlled adversarial conditions using **Foolbox**:

| Attack   | Norm | Description                                   |
| -------- | ---- | --------------------------------------------- |
| FGSM     | L2   | Single-step gradient-based attack             |
| iFGSM    | L2   | Iterative FGSM (multi-step PGD variant)       |
| DeepFool | L2   | Finds minimal boundary-crossing perturbations |

Each attack is evaluated across multiple normalized L2 radii, and robustness curves are plotted for direct comparison.

---

## Execution Summary

1. Train the baseline and RaCNN models on CIFAR-10.
2. Build the FAISS feature index for retrieval.
3. Evaluate both models:

   * Clean accuracy
   * FGSM (L2)
   * iFGSM (L2)
   * DeepFool (L2)
4. Generate robustness plots comparing baseline vs. RaCNN performance.

All stages are contained in the `racnn_implement.ipynb` notebook.

---

## Citation

Zhao, J., & Cho, K. (2018). *Retrieval-Augmented Convolutional Neural Networks against Adversarial Examples.*
NeurIPS Workshop on RobustML. [arXiv:1802.09502](https://arxiv.org/abs/1802.09502)

---

