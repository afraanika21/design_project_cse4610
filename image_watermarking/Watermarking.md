
---

# Watermark-Embedded Adversarial Examples for Copyright Protection

## Overview

This repository re-implements the framework proposed in ***Watermark-Embedded Adversarial Examples for Copyright Protection against Diffusion Models*** by Peifei Zhu et al. (CVPR 2024).
The system trains a **conditional GAN generator** that embeds a personal watermark into adversarial perturbations.
These perturbations cause diffusion models (e.g., Stable Diffusion 1.5) to generate **watermarked or chaotic outputs**, thus preventing unauthorized imitation of copyrighted artwork.

## Core Idea

* Build a generator G and discriminator D using a **conditional GAN** architecture.
* G receives an image x and watermark m and predicts a perturbation δ = G(x | m).
* The adversarial example is `x′ = x + δ · PERT_BOUND`.
* A pretrained Stable Diffusion VAE is frozen to compute latent-space adversarial loss.

### Loss Components

| Loss                 | Purpose                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------ |
| **L<sub>GAN</sub>**  | Forces generated images to remain perceptually close to originals                          |
| **L<sub>pert</sub>** | Weighted soft-hinge penalty constraining perturbation magnitude                            |
| **L<sub>adv</sub>**  | Latent-space adversarial loss encouraging diffusion models to reproduce watermark features |

Final objective:
[
\min_G \max_D ; L_{adv} + \alpha L_{GAN} + \beta L_{pert}
]
with α = 1.0 and β = 10.0.

---

## Dataset

We follow the experimental design in the paper using **WikiArt** paintings for artist-specific watermarks.

* **Source**: [WikiArt Dataset](https://www.wikiart.org/) (local path `E:/wikiart_dataset`)
* **Structure**: Each filename encodes artist → `artist-name_title.jpg`
* **Watermarks**: Auto-generated per artist (e.g., `BEATTIE`, `MONET`, `REMBRANDT`)
* **Training Split**: 50 artists × 10 images each
* **Validation Split**: Remaining images
* **Image Size**: 256 × 256 pixels

Watermarks are rendered with `PIL.ImageDraw` in binary (white text on black background).

---

## Environment Setup

### Dependencies

```bash
pip install pillow
pip install diffusers transformers tokenizers==0.20.1 accelerate==1.0.1 safetensors==0.4.5 huggingface_hub==0.25.2
pip install opencv-python scikit-image pandas matplotlib tqdm
```

### Hardware

* Python ≥ 3.10
* PyTorch ≥ 2.0 with CUDA support
* GPU recommended (≥ 8 GB VRAM)

The model can fall back to CPU execution if CUDA is unavailable.

---

## Directory Layout

```
watermarking/
│
├── watermarking.ipynb          # Full training + evaluation notebook
├── output/
│   ├── generator_best.pth      # Best generator weights
│   ├── preview_epochXX.png     # Periodic training previews
│   ├── orig_samples.png        # Original validation batch
│   ├── adv_samples.png         # Adversarially watermarked samples
│   └── gen_from_adv.png        # SD outputs showing visible watermark
└── E:/wikiart_dataset/         # Local dataset root (user-defined)
```

---

## Model Components

### Generator G

* Encoder–decoder ResNet architecture
* Input channels = 4 (RGB + mask)
* Output = 3-channel perturbation map ∈ [-1, 1]
* Four residual blocks at ¼ resolution

### Discriminator D

* Multi-layer CNN with instance norm and LeakyReLU
* Outputs scalar probability (real vs fake)

### Stable Diffusion VAE

Loaded via:

```python
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
```

Used only for computing latent L<sub>adv</sub>.

---

## Training Configuration

| Parameter      | Value      | Description                    |
| -------------- | ---------- | ------------------------------ |
| IMG_SIZE       | 256        | Input resolution               |
| BATCH_SIZE     | 8          | Mini-batch size                |
| EPOCHS         | 200        | Training epochs                |
| LR             | 1e-3       | Learning rate (Adam)           |
| PERT_BOUND     | 10 / 255   | Maximum perturbation magnitude |
| W<sub>WM</sub> | 4.0        | Watermark weighting            |
| α , β          | 1.0 , 10.0 | Loss coefficients              |

Training time ≈ 20 min on RTX A100 for 100 samples (0.2 s per image generation after training).

---

## Evaluation Pipeline

1. **Generate Adversarial Examples**

   * Apply trained G to test images to produce `adv = x + δ · PERT_BOUND`.
2. **Feed into Diffusion Model**

   * Run Stable Diffusion img2img generation with prompt (e.g., “A painting”, strength = 0.3).
3. **Compute Metrics**

   * MSE, PSNR, SSIM → image quality
   * FID, Precision, Recall → diffusion outputs
   * NCC → watermark visibility
4. **Visualize**

   * Side-by-side plots (original / adversarial / generated / watermark)

Example metrics (WikiArt @ strength 0.3):

| Method   | MSE ↓  | PSNR ↑ | SSIM ↑ | NCC ↑    | Runtime (s) |
| -------- | ------ | ------ | ------ | -------- | ----------- |
| AdvDM    | 0.0038 | 29.1   | 0.80   | 0.00     | 32          |
| Mist     | 0.0040 | 29.0   | 0.81   | 0.09     | 35          |
| **Ours** | 0.0037 | 30.1   | 0.80   | **0.31** | **0.2**     |

---

## Inference Example

```python
# Load best model
G.load_state_dict(torch.load("output/generator_best.pth", map_location=device))
G.eval()

# Produce adversarial sample
img, wm = next(iter(val_loader))
adv = torch.clamp(img + G(img, wm) * PERT_BOUND, 0, 1)
```

Then use `StableDiffusionImg2ImgPipeline` to visualize watermark transfer.

---

## Expected Results

* Diffusion outputs display clear textual watermarks (e.g., artist names).
* Adversarial perturbations remain visually imperceptible.
* Training generator with 10 samples per artist is sufficient for robust watermark embedding.
* Demonstrates cross-model transferability to DreamBooth, LoRA, and NovelAI.

---

## Citation

If you use this implementation, please cite the original paper:

```
@inproceedings{zhu2024watermarkadversarial,
  title={Watermark-Embedded Adversarial Examples for Copyright Protection against Diffusion Models},
  author={Peifei Zhu and Tsubasa Takahashi and Hirokatsu Kataoka},
  booktitle={CVPR},
  year={2024}
}
```
