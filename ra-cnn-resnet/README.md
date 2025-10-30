# 🧠 Retrieval-Augmented Convolutional Neural Network (RaCNN-ResNet)

This notebook reproduces and extends the core ideas from  
**Cho et al., *Retrieval-Augmented Convolutional Neural Networks against Adversarial Examples*, NeurIPS 2019.**

RaCNN combines a convolutional network with a retrieval engine to improve robustness against both **off-manifold** and **on-manifold** adversarial examples.  
The retrieval engine locally approximates the data manifold via a **feature-space convex hull**, while the classifier is trained with **local mixup** regularization to encourage linear behavior inside that region.

---

## 📘 Overview

This experiment implements the full RaCNN pipeline using **ResNet-18** on **CIFAR-10**, trained and evaluated in Google Colab.

**Main components implemented (from the paper):**
| Paper Section | Component | Implementation Detail |
|----------------|------------|------------------------|
| § 2.1 | Local characterization of the data manifold | FAISS-based K-nearest neighbor retrieval on features φ′(x) |
| § 2.1 | Trainable projection onto convex hull | Attention weights αₖ = softmax(φ(xₖ′)ᵀ U φ(x)) |
| § 2.2 | Local Mixup | Kraemer-style convex combination of retrieved neighbors |
| § 2.3 | Retrieval Engine F | FAISS IndexFlatL2 over 50 k CIFAR-10 features (φ′ frozen ResNet-18) |
| § 6 | Evaluation under Adversarial Attacks | FGSM, iFGSM, PGD, DeepFool, CW, and Noise perturbations |

---

## ⚙️ Implementation Details

- **Feature extractors**
  - φ′ (“phi-prime”): frozen pretrained ResNet-18 (ImageNet)  
  - φ (trainable): ResNet-18 initialized from ImageNet weights  
- **Retrieval engine F:** FAISS IndexFlatL2 (K = 10 neighbors)  
- **Classifier g:** two-layer MLP trained jointly with φ and projection U  
- **Training:**
  - Optimizer: SGD (momentum 0.9, lr 1e-3, weight decay 5e-4)  
  - Epochs: 20 (on Colab T4 GPU)  
  - Loss: Cross-Entropy + Local Mixup regularization  
- **Dataset:** CIFAR-10 (50 k train / 10 k test, 224×224 resized)

---
---

## 🔍 Comparison with Original Paper

| Aspect | Original (Cho et al., 2019) | This Implementation | Rationale |
|--------|-----------------------------|---------------------|------------|
| **Backbone** | Custom 6-layer CNN | **ResNet-18 (ImageNet pretrained)** | Easier to reproduce and faster convergence in Colab |
| **Feature extractor φ′** | CNN without final FC layer | **Frozen ResNet-18 encoder** | Stronger and more general features improve retrieval quality |
| **Retrieval engine F** | Locality-Sensitive Hashing (LSH) | **FAISS IndexFlatL2 (exact dense retrieval)** | Simpler to use, GPU-accelerated, scales to 50k examples |
| **Projection matrix U** | Trainable attention matrix | ✅ Same (softmax(φ(xₖ′)ᵀUφ(x))) | Faithfully implemented |
| **Local mixup** | Kraemer algorithm | ✅ Same, implemented in PyTorch | Fully matches paper design |
| **Dataset size** | CIFAR-10 (32×32) | **CIFAR-10 resized to 224×224** | Required for pretrained ResNet input size |
| **Training steps** | SGD (learning rate 0.1) | **SGD (lr = 1e-3)** | Lower learning rate for pretrained model fine-tuning |
| **Adversarial evaluation** | FGSM, iFGSM, DeepFool, CW, L-BFGS, Boundary | **FGSM, iFGSM, PGD, DeepFool, CW, Noise** | Boundary and L-BFGS excluded due to Colab time/memory limits |
| **Hardware** | GPU (likely Tesla V100) | **Free Colab T4 (16 GB)** | Adjusted batch size and epochs for free-tier runtime |

> ⚖️ Despite these simplifications, the reproduced RaCNN achieves comparable robustness trends, validating the retrieval-augmented defense concept under modern architectures.

---

## ▶️ How to Run on Colab

1. **Open the notebook** `RaCNN-ResNet.ipynb` in Google Colab.  
2. **Enable GPU:** Runtime → Change runtime type → GPU (T4 recommended).  
3. **Run all cells sequentially.**  
   - CIFAR-10 is automatically downloaded.  
   - FAISS builds the retrieval index on first run (≈ 1 min).  
4. **Outputs:** Model checkpoints (`.pth`), feature DB (`.npy` + `.faiss`), and printed attack accuracies.

---


> 🧩 This experiment was reproduced for research understanding only.  
> For future updates (RaCNN-ViT, AutoMixup), check the main project README.
```
