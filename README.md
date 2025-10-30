

---

# Design Project — CSE4610

### Implementation and Experimental Analysis of Modern Adversarial Learning and Vision Models

This repository contains the collective work of **Team CSE4610**, where we implemented, analyzed, and partially reproduced several **research papers in adversarial machine learning, vision-language modeling, and diffusion-based robustness**.
The project is organized into distinct modules reflecting each team member’s research focus and implementation.

---

##  Project Overview

| Member     | Research Papers Implemented                                                                                                                                                                                                                                                                                     | Key Modules                                                     |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Aimaan** | 1️. *Large-scale Classification of Fine-Art Paintings: Learning the Right Metric on the Right Feature*  <br>2️. *VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models* (partial)                                                                                            | `artwork_classification/`, `vlattack/`                          |
| **Afra**   | 1️. *Explaining and Harnessing Adversarial Examples* (Goodfellow et al., ICLR 2015)  <br>2️. *Retrieval-Augmented Convolutional Neural Networks against Adversarial Examples* (Zhao & Cho, ICLR 2021) using ResNet18                                                                                            | `basic_practice_advAttack/`, `ra-cnn-resnet/`, `ra-cnn-custom/` |
| **Tanjil** | 1️. *Fine-tuning Segment Anything Model (SAM) for Industrial Defect Detection*  <br>2️. *Watermark-Embedded Adversarial Examples for Copyright Protection against Diffusion Models* (Zhu et al., CVPR 2024)  <br>3️. *Retrieval-Augmented Convolutional Neural Networks* (RaCNN) with Foolbox attack evaluation | `fine_tuning_SAM/`, `image_watermarking/`, `racnn_foolbox/`     |

---

##  Research Motivation

Adversarial learning has become a cornerstone for evaluating and enhancing the **robustness, interpretability, and copyright protection** of deep models.
Our goal was to **implement and compare real-world methods** from top-tier research to understand:

* How adversarial perturbations affect different modalities (vision, vision-language).
* How retrieval-based and manifold projection defenses (RaCNN) mitigate such attacks.
* How watermark embedding and diffusion-based protection can secure creative content.
* How pre-trained models like SAM and BLIP behave under fine-tuning or attacks.

---

## 📂 Repository Structure

```
DESIGN_PROJECT_CSE4610/
│
├── artwork_classification/       # Aimaan – Fine-Art painting classification & metric learning
│   ├── Artwork_Classification.ipynb
│   ├── artwork.md
│
├── basic_practice_advAttack/     # Afra – FGSM, meme classifier robustness
│   ├── fgsm_meme_classifier.ipynb
│   ├── README.md
│
├── fine_tuning_SAM/              # Tanjil – Fine-tuning SAM for industrial defects
│   ├── fine-tune-sam-for-industrial-defect.ipynb
│   ├── Fine_Tune_SAM.md
│
├── image_watermarking/           # Tanjil – Watermark adversarial examples (CVPR 2024)
│   ├── watermarking.ipynb
│   ├── Watermarking.md
│
├── ra-cnn-custom/                # Afra – RaCNN with vanilla CNN & retrieval augmentation
│   ├── ra-cnn-custom.ipynb
│   ├── config.json
│   ├── readme.md
│
├── ra-cnn-resnet/                # Afra – RaCNN with ResNet18 backbone
│   ├── RA_CNN_resnet18.ipynb
│   ├── README.md
│
├── racnn_foolbox/                # Tanjil – RaCNN defense + Foolbox adversarial attacks
│   ├── racnn_implement.ipynb
│   ├── RACNN_implement.md
│
├── vlattack/                     # Aimaan – Vision-language adversarial attack (VLAttack)
│   ├── VLAttack_on_CLIP.ipynb
│   ├── vlattack.md
│
├── LICENSE
└── .gitignore
```

---

## 🔬 Research Implementations

###  Artwork Classification (Aimaan)

**Paper:** *Large-scale Classification of Fine-Art Paintings: Learning the Right Metric on the Right Feature*
**Objective:** Evaluate CNN + metric learning methods (contrastive / triplet) on art styles and genres.
**Dataset:** [WikiArt Dataset](https://www.wikiart.org/)
**Key Features:**

* Feature extraction using VGG16 / ResNet encoders
* Metric learning with triplet loss
* Cross-artist generalization analysis

###  VLAttack on CLIP (Aimaan)

**Paper:** *VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models*
**Objective:** Examine multimodal attacks that disrupt CLIP’s vision-language alignment.
**Implementation:**

* Attack on BLIP/CLIP image-text retrieval
* Perturbations transferred across modalities
* Evaluation on small sample set for feasibility

---

### Adversarial Examples & RaCNN (Afra)

**Papers:**

* *Explaining and Harnessing Adversarial Examples* (Goodfellow et al., 2015)
* *Retrieval-Augmented Convolutional Neural Networks against Adversarial Examples* (Zhao & Cho, 2021)

**Objective:** Study adversarial generation (FGSM, iFGSM) and retrieval-based defense architectures.
**Key Work:**

* Built FGSM-based meme classifier robustness test
* Implemented RaCNN using ResNet18 feature extractor
* Created variant RaCNN with configurable retrieval layers and local mixup

---

###  SAM Fine-tuning for Industrial Defects (Tanjil)

**Objective:** Adapt the *Segment Anything Model (SAM)* for fine-grained defect detection.
**Dataset:** [Few-Shot Industrial Defect Detection (Kaggle)](https://www.kaggle.com/datasets/aryashah2k/few-shot-industrial-defect-detection)
**Key Methods:**

* Fine-tuned SAM ViT-B encoder
* Evaluated IoU improvement for “good” vs “bad” class defects
* Integrated efficient annotation and masking visualization

---

###  Watermark-Embedded Adversarial Examples (Tanjil)

**Paper:** *Watermark-Embedded Adversarial Examples for Copyright Protection against Diffusion Models* (CVPR 2024)
**Objective:** Train a conditional generator to embed artist-specific invisible watermarks in artwork that remain traceable in Stable Diffusion outputs.
**Key Work:**

* Implemented GAN-based perturbation generator
* Used Stable Diffusion v1.5 VAE for latent-space adversarial loss
* Measured watermark persistence via NCC and diffusion response

---

###  RaCNN Defense with Foolbox (Tanjil)

**Paper:** *Retrieval-Augmented Convolutional Neural Networks against Adversarial Examples*
**Objective:** Extend RaCNN defense analysis with Foolbox adversarial attacks.
**Techniques:**

* Implemented FGSM, iFGSM, PGD attacks via Foolbox
* Compared baseline CNN vs RaCNN robustness
* Evaluated performance under L∞ and L2 constraints

---

##  Environment Summary

| Library       | Version |
| ------------- | ------- |
| PyTorch       | ≥ 2.0   |
| torchvision   | ≥ 0.15  |
| diffusers     | ≥ 0.29  |
| transformers  | ≥ 4.43  |
| pandas        | ≥ 2.0   |
| numpy         | ≥ 1.25  |
| matplotlib    | ≥ 3.7   |
| foolbox       | ≥ 3.3   |
| opencv-python | ≥ 4.8   |

Each notebook includes its own environment setup cell for reproducibility (Kaggle or Colab compatible).

---

##  License

This repository is distributed for **academic and educational use** under the MIT License.
Please cite the respective original papers when reusing or extending these implementations.

---

##  Contributors

| Name       | ID | Contribution                                          |
| ---------- | -- | ----------------------------------------------------- |
| **Aimaan Ahmed** | 210041204  | Fine-Art classification, VLAttack                     |
| **Afra Anika**   | 210041206 | Adversarial training, RaCNN-ResNet                    |
| **Tanjil Hasan Khan** | 210041246  | SAM fine-tuning, Watermark adversarial, RaCNN defense |

---
