
***

# README for VLAttack on CLIP Notebook

## Overview

This notebook presents an extensive exploration and adversarial attack strategy on the CLIP (Contrastive Language-Image Pretraining) model. The main goals were:

- To understand CLIP's robustness against adversarial modifications.
- To generate adversarial examples targeting CLIP's image and text modalities.
- To analyze the transferability of these attacks and evaluate CLIP’s vulnerabilities.

Throughout the notebook, various attack methods and perturbation techniques are implemented, tested, and evaluated against CLIP's image and text encoders.

***

## What Was Done

- **Data Preparation:**  
  Loaded and preprocessed images and text prompts for targeted attacks.

- **Adversarial Attack Implementation:**  
  Developed multiple attack algorithms, such as PGD, FGSM, and iterative optimization, to generate adversarial examples for both image and text modalities within CLIP.

- **Model Testing:**  
  Evaluated the attack success rates, transferability between different samples, and the robustness of CLIP’s multimodal embeddings.

- **Experiment Challenges:**  
  Attempted to fine-tune the CLIP model to enhance attack success or explore robustness further. However, **the entire experiment could not be completed** due to **memory limitations when attempting to fine-tune CLIP** on the dataset.

***

## Libraries and Dependencies

The experiments in this notebook relied heavily on the following libraries:

- **`torch`** and **`transformers`** (from Hugging Face): For loading the CLIP model and tokenizers.
- **`tqdm`**: For progress bar visualization.
- **`numpy`** and **`scipy`**: For numerical computations.
- **`PIL` (Pillow)**: For image manipulation.
- **`scikit-learn`**: For evaluation metrics.
- **`torchattacks`** (or custom implementations): For adversarial attack algorithms.

**Note:** Since the experiment with CLIP fine-tuning was halted due to **excessive memory use**, it was not completed.

***

## Important Note
The entire experiment, including fine-tuning CLIP, **could not be fully executed** due to **insufficient memory resources** on the hardware used, which prevented long or large-scale fine-tuning experiments.

***

## Usage
Clone this repository, install the dependencies listed above, and run the notebook to replicate or extend the attack experiments on CLIP.

***
