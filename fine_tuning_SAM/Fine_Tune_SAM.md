## Fine-Tuning Segment Anything (ViT-B) for Industrial Defect Segmentation

This repository fine-tunes Meta’s Segment Anything Model (SAM) on the **Few-Shot Industrial Defect Detection** dataset to perform defect localization and segmentation for products such as capsules, pills, and bottles.
Only the **mask decoder** of SAM is trained while keeping the encoders frozen.

---

## Overview

* **Base Model:** Segment Anything (ViT-B)
* **Dataset:** Few-Shot Industrial Defect Detection (Kaggle)
* **Task:** Binary segmentation of defective regions
* **Training Target:** Mask decoder only
* **Prompt Type:** Bounding-box prompts derived from ground-truth masks
* **Frameworks:** PyTorch, Albumentations, MONAI, Hugging Face Transformers

---

## Dataset Structure

Each category folder contains `good` and `bad` subfolders:

```
dataset/
├── capsule/
│   ├── good/
│   │   ├── 000.png
│   │   └── ...
│   ├── bad/
│       ├── test_crack_021.png    ← defective image
│       ├── test_crack_021.bmp    ← binary mask
│       └── ...
└── pill/
    ├── good/
    ├── bad/
```

* `.png` files are RGB input images.
* `.bmp` files are grayscale defect masks.

---

## Environment Setup

Install dependencies:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/huggingface/transformers.git
pip install albumentations monai datasets opencv-python scikit-learn tqdm matplotlib
```

Tested on:
Kaggle GPU (T4, CUDA 11.8, Python 3.10, PyTorch 2.1)

---

## Project Files

| File                 | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `sam_finetune.ipynb` | Main notebook for preprocessing, training, and evaluation |
| `SAMDefectDataset`   | PyTorch dataset class for SAM-style input                 |
| `get_bounding_box()` | Computes bounding box prompts from binary masks           |
| `compute_dice()`     | Computes Dice similarity for evaluation                   |
| `infer_and_plot()`   | Visualizes predictions against ground truth               |

---

## Model Setup

```python
from segment_anything import sam_model_registry

checkpoint_path = "/kaggle/input/segment-anything/pytorch/vit-b/1/model.pth"
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to("cuda")

# Freeze encoders
for p in sam.image_encoder.parameters():
    p.requires_grad = False
for p in sam.prompt_encoder.parameters():
    p.requires_grad = False
```

---

## Data Preparation

```python
samples = get_samples("/kaggle/input/few-shot-industrial-defect-detection", categories=["capsule", "pill"])
train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)

train_dataset = SAMDefectDataset(train_samples, transform=train_transform)
test_dataset = SAMDefectDataset(test_samples)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
```

Data augmentation with Albumentations includes flips, rotations, shifts, and brightness/contrast adjustments.

---

## Training Configuration

* **Optimizer:** Adam (lr = 1e-4)
* **Loss:** DiceCELoss (sigmoid = True)
* **Scheduler:** ReduceLROnPlateau
* **Epochs:** 30
* **Batch Size:** 1
* **Training Target:** Mask decoder parameters only

Example loop:

```python
for epoch in range(num_epochs):
    sam.train()
    for batch in train_loader:
        image = batch["image"].to("cuda")
        mask_gt = batch["mask"].to("cuda")
        box = batch["box"].to("cuda").unsqueeze(0)

        image_embedding = sam.image_encoder(image)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(boxes=box, points=None, masks=None)
        image_pe = sam.prompt_encoder.get_dense_pe().to(image.device)

        low_res_masks, _ = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        pred_mask = F.interpolate(low_res_masks, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss_fn(pred_mask, mask_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Save the fine-tuned decoder:

```python
torch.save(sam.mask_decoder.state_dict(), "/kaggle/working/sam_finetuned_box_only.pth")
```

---

## Evaluation

Compute Dice score:

```python
dice = compute_dice(pred_mask_bin.cpu().numpy(), mask_gt.cpu().numpy())
```

Visualize predictions:

```python
infer_and_plot(
    image_path="/kaggle/input/few-shot-industrial-defect-detection/capsule/bad/test_crack_021.png",
    mask_path="/kaggle/input/few-shot-industrial-defect-detection/capsule/bad/test_crack_021.bmp"
)
```

Outputs:

* Left: Input image
* Middle: Predicted mask
* Right: Ground truth mask

---

## Results Summary

| Category | Validation Dice (avg.) | Observation                               |
| -------- | ---------------------- | ----------------------------------------- |
| Capsule  | ~0.82                  | Good localization of cracks and scratches |
| Pill     | ~0.79                  | Accurate for edge defects                 |
| Average  | ~0.80                  | Stable after ~25 epochs                   |

---

## Checkpoints

| File                                           | Description                     |
| ---------------------------------------------- | ------------------------------- |
| `model.pth`                                    | Pretrained SAM ViT-B checkpoint |
| `sam_finetuned_box_only.pth`                   | Fine-tuned mask decoder weights |
| `sam_decoder_finetuned_medicine_corrected.pth` | Refined checkpoint              |

---

## Future Improvements

* Add point or mask prompts for hybrid prompting.
* Fine-tune encoders with smaller learning rate.
* Evaluate cross-category generalization.
* Integrate inference into a web interface.

---

**Author:** Tanjil Hasan Khan
**Date:** August 2025
**Environment:** Kaggle Notebook, T4 (16 GB)
---------------------------------------------------
