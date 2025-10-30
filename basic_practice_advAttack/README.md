
# FGSM Attack on BLIP Image Captioning Model

This notebook demonstrates a simple **adversarial attack** on a **vision-language model** — specifically, the **BLIP** image captioning model.  
It shows how a small perturbation using **Fast Gradient Sign Method (FGSM)** can change the generated caption for the same image.

Source notebook: `fgsm_meme_classifier.ipynb` 【11†file】

---

## Overview

Steps performed by the notebook:

1. Load the pre-trained BLIP model and processor from Hugging Face.
2. Upload an image via Google Colab interface.
3. Generate a caption using BLIP for the clean/original image.
4. Apply **FGSM** attack to perturb the image pixels based on model gradients.
5. Generate a new caption from the perturbed image.
6. Display both images side-by-side to visualize the effect.

---

## Installation

Install required packages (notebook cell includes this):

```bash
!pip install torch torchvision transformers pillow matplotlib
```

---

## Notebook Structure

- **Imports**  
  - `transformers` (BlipProcessor, BlipForConditionalGeneration)  
  - `PIL` (Image)  
  - `torch`, `torchvision.transforms`, `matplotlib.pyplot`

- **Model loading**
```python
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()
```

- **Image upload (Colab)**
```python
from google.colab import files
uploaded = files.upload()
```

- **Caption generation**
```python
def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
```

- **FGSM attack**
```python
def fgsm_attack(image_path, epsilon=0.03):
    image = Image.open(image_path).convert("RGB")
    original_caption = generate_caption(image_path)
    inputs = processor(images=image, text=original_caption, return_tensors="pt")
    inputs['pixel_values'].requires_grad = True
    inputs['labels'] = inputs['input_ids']

    model.train()
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()

    perturbed = inputs['pixel_values'] + epsilon * inputs['pixel_values'].grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)

    attacked_img = transforms.ToPILImage()(perturbed.squeeze())
    return attacked_img
```

- **Main processing loop**
  - For each uploaded image: generate original caption, run FGSM, save attacked image as `attacked_<original_filename>`, generate attacked caption, and show original + attacked images side-by-side.

---

## Usage

1. Open the notebook in Google Colab.
2. Run the first cell to install dependencies.
3. Run model-loading cells (they will download BLIP weights).
4. Use the Colab file upload widget to upload one or more images.
5. Run the attack cell to produce `attacked_<image>` files and visualize results.

---

## Parameters

- `epsilon` — controls FGSM strength. Typical values: `0.01` to `0.1`. The notebook uses `epsilon=0.03`.
- Larger epsilon → stronger perturbation → more likely to change captions, but may introduce visible artifacts.

---

## Outputs

- `attacked_<image_name>`: saved perturbed image file.
- Console outputs showing:
  - Original caption
  - Loss before attack
  - Caption after attack
- Side-by-side figure displaying original and attacked images.

---

## Limitations & Notes

- This notebook uses the original caption as a pseudo-label for the loss; this is a heuristic and not guaranteed to be the strongest adversarial objective for miscaptioning.
- The attack shown is a single-step FGSM. Iterative methods (I-FGSM, PGD) are typically stronger.
- The processor and model expect specific preprocessing; ensure `pixel_values` are in the expected range (the notebook clamps to `[0, 1]`).
- Running gradient-based attacks requires enough memory; Colab free GPUs may be limited.

---

## Suggested Extensions

- Implement iterative attacks: I-FGSM, PGD.
- Use alternative objectives (e.g., maximize negative log-likelihood of the original caption, or target a specific wrong caption).
- Quantify caption change with automatic metrics (BLEU, ROUGE, CIDEr) between original and attacked captions.
- Visualize perturbation map (difference between original and attacked images).
- Test on other captioning models (BLIP-2, OFA, GIT).

---

## References

- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR.
- Li, J., Li, D., Savarese, S., & Hoi, S. (2022). *BLIP: Bootstrapped Language-Image Pre-training*. arXiv.

---

## License

Use and modify freely for research and education. Cite the notebook and the BLIP model if used in publications.

