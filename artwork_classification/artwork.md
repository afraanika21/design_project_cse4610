

***

# README for Artwork Classification Notebook

## Overview

This notebook implements a large-scale classification pipeline for artwork images using the WikiArt dataset. The primary goal was to recreate the methodology from the Saleh & Elgammal (2015) paper on learning optimized similarity metrics and features for classifying paintings by **style**, **genre**, and **artist**.

***

## What Was Done

- **Dataset Loading:** Utilized the Hugging Face `huggan/wikiart` dataset, loading subsets to manage memory limits.
- **Feature Extraction:** Extracted **Classeme features** using a pretrained deep CNN (VGG16’s intermediate layers), capturing semantic object presence information in images.
- **Dimensionality Reduction:** Applied PCA to reduce Classeme's high-dimensional features to 512 dimensions for efficiency.
- **Metric Learning:** Implemented multiple metric learning algorithms (LMNN, ITML, NCA, MLKR) to learn improved similarity metrics for the artwork features.
- **Classification:** Trained linear SVM classifiers on both raw and metric-learned projected features to predict painting style (default), genre, or artist labels.
- **Feature Fusion:** Provided capability to combine multiple metric projections for potentially improved classification performance.
- **Evaluation:** Conducted accuracy evaluations showing improvements with metric learning compared to baseline classification on Classeme features.
- **Visualization:** Included image display with labels to verify dataset correctness.

***

## Special Library Requirements

- `numpy`
- `tensorflow` (for VGG16 pretrained CNN and feature extraction)
- `datasets` (Hugging Face datasets for WikiArt loading)
- `Pillow` (PIL for image preprocessing)
- `scikit-learn` (PCA, SVM, metric learning support)
- `metric-learn` (for LMNN, ITML, NCA, MLKR implementations)
- `tqdm` (progress bars)
- Optionally: `joblib` for saving/loading models

***

## Limitations and Notes

- The pipeline focuses on **Classeme features** as these gave results closest to the original paper’s benchmarks.
- Due to **hardware memory constraints**, the entire end-to-end experiment, especially fine-tuning CLIP or very large-scale training, **could not be completed**.
- The approach uses subset sampling and batch-wise incremental feature extraction to manage Colab resource limits.
- MLKR sometimes exhibited long run times; it was either limited in iterations or skipped.
- Results are consistent with the Saleh & Elgammal findings: metric learning improves classification accuracy over raw features.

***

