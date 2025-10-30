
# 🧠 Retrieval-Augmented Convolutional Neural Networks (RaCNN)
### *Robustness Evaluation under Adversarial Attacks*

This repository implements and extends the paper:

> **Retrieval-Augmented Convolutional Neural Networks for Improved Robustness against Adversarial Examples**  
> *Jake (Junbo) Zhao and Kyunghyun Cho, ICLR 2019*

We reproduce the *retrieval-based adversarial defense* pipeline using **CIFAR-10**,  
train baseline and RaCNN variants (K = 5, 10) with **local mixup**,  
and evaluate robustness under **FGSM**, **iFGSM**, and **PGD** attacks  
using both **TorchAttacks** and **Foolbox** frameworks.

---

## 📁 Project Structure

```

ra-cnn-custom/
│
├── checkpoints/                     # Saved model weights (.pt)
│   ├── baseline_best.pt
│   ├── racnn_k5_best.pt
│   └── racnn_k10_best.pt
│
├── artifacts/                       # Output results & plots
│   ├── results_torchattacks_summary.json
│   ├── scenario1_curves.json
│   ├── scenario1_attack_curves_final.png
│   ├── scenario2_foolbox_curves.png
│   ├── scenario2_fgsm_racnn_k10.json
│   ├── scenario2_ifgsm_racnn_k10.json
│   └── scenario2_*_racnn_k10.png
│
├── main_notebook.ipynb              # Full implementation & experiments
└── README.md

````

---

## ⚙️ Environment Setup

Install all dependencies:

```bash
pip install torch torchvision tqdm matplotlib numpy
pip install torchattacks foolbox
````

> ✅ Tested with: `torch==2.3+`, `torchattacks==3.5`, `foolbox==3.3`, `CUDA 12.8`.

---

## 🧩 1. Dataset and Preprocessing

We use **CIFAR-10** with standard normalization:

[
\text{mean} = (0.4914, 0.4822, 0.4465), \quad
\text{std} = (0.2023, 0.1994, 0.2010)
]

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])
train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
test_loader  = DataLoader(testset, batch_size=256)
```

---

## 🧱 2. Model Architecture

### 🟦 Baseline CNN

A simple convolutional network trained with cross-entropy loss.

### 🟩 Retrieval-Augmented CNN (RaCNN)

RaCNN augments the CNN backbone with a **retrieval engine** that projects
inputs onto the local data manifold before classification.

* Retrieves *K nearest neighbors* from a FAISS feature store.
* Combines query and neighbor features using **local mixup**:
* Mixup coefficients dynamically sampled per batch.

We implemented both:

* **RaCNN K = 5**
* **RaCNN K = 10**

---

## 🎯 3. Training Configuration

|           Parameter |                               Value |
| ------------------: | ----------------------------------: |
|           Optimizer | SGD (lr = 0.01) or Adam (lr = 1e-3) |
|        Weight Decay |                            5 × 10⁻⁴ |
|          Batch Size |                                 128 |
|              Epochs |                                  30 |
| NMU (mixup updates) |                         5 per batch |
|      Early Stopping |                        patience = 6 |
|                 AMP |                   ✅ Mixed Precision |

Checkpoints are automatically saved to `checkpoints/`.

---

## 🧪 4. Scenario 1 — Direct Adversarial Attack

This corresponds to **Section 6, Scenario 1** of the paper.
We directly attack the *visible classifier* using **TorchAttacks**.

### Implemented Attacks

| Attack | Library             | Description                | Steps |
| :----- | :------------------ | :------------------------- | :---- |
| FGSM   | `torchattacks.FGSM` | Single-step sign-gradient  | 1     |
| iFGSM  | `torchattacks.PGD`  | Iterative variant of FGSM  | 10    |
| PGD    | `torchattacks.PGD`  | Stronger multi-step attack | 20    |

Each attack is evaluated on:

* **Baseline CNN**
* **RaCNN-K5 (with mixup)**
* **RaCNN-K10 (with mixup)**

### Output Metrics

* **Accuracy (%)**
* **Mean normalized L₂ dissimilarity**

[
L_2^{norm} = \frac{||x - x_{adv}||_2^2}{\text{dim}(x)}
]

### Output Files

* `scenario1_curves.json`
* `scenario1_attack_curves_final.png`

Example curves:

```
FGSM → Accuracy vs. Normalized L2
iFGSM → Accuracy vs. Normalized L2
PGD → Accuracy vs. Normalized L2
```

---

## 🧠 5. Scenario 2 — Hidden Retrieval Attack (Foolbox)

In this scenario, the attacker can access the **feature extractor and classifier**,
but **not** the retrieval engine.
We simulate this setting with **Foolbox** using L₂ PGD attacks.

### Attacks Implemented

| Attack | Library | Bound | Steps | Description                        |
| :----- | :------ | :---- | :---- | :--------------------------------- |
| L₂ PGD | Foolbox | L₂    | 40    | Standard projected gradient attack |
| FGSM   | Foolbox | L₂    | 1     | Fast gradient single step          |
| iFGSM  | Foolbox | L₂    | 10    | 10-step iterative variant          |


### Outputs

* `scenario2_foolbox_curves.png` — Baseline vs RaCNN curves
* `scenario2_fgsm_racnn_k10.json` / `.png`
* `scenario2_ifgsm_racnn_k10.json` / `.png`

---

## 📈 6. Results Summary

| Model        | Clean Acc | FGSM (ε = 8/255) | iFGSM (10 steps) | PGD (20 steps) |
| :----------- | :-------: | :--------------: | :--------------: | :------------: |
| Baseline CNN |   85.6 %  |      30.9 %      |       8.1 %      |      7.6 %     |
| RaCNN K = 5  |   86.5 %  |      47.0 %      |      45.7 %      |     45.8 %     |
| RaCNN K = 10 |   87.2 %  |      49.3 %      |      47.6 %      |     47.5 %     |

---

## 🧾 7. Generated Plots

| Figure                                | Description               | File                                |
| :------------------------------------ | :------------------------ | :---------------------------------- |
| Accuracy vs. Norm-L₂ (FGSM/iFGSM/PGD) | Scenario 1 (TorchAttacks) | `scenario1_attack_curves_final.png` |
| Robustness (L₂ PGD – Foolbox)         | Scenario 2                | `scenario2_foolbox_curves.png`      |
| FGSM Robustness (K10 only)            | Foolbox FGSM              | `scenario2_fgsm_racnn_k10.png`      |
| iFGSM Robustness (K10 only)           | Foolbox PGD(10)           | `scenario2_ifgsm_racnn_k10.png`     |

---

## 🧮 8. Normalized L₂ Metric

For every adversarial example pair ((x, x_{adv})):

[
L_2^{norm} = \frac{||x - x_{adv}||_2^2}{3072}
]

This normalization ensures direct comparison across images
and aligns with Section 5.1 of the RaCNN paper.

---

## 🧰 9. Key Implementation Details

* **Mixed Precision (AMP)** used for efficient training.
* **Early Stopping** prevents overfitting (patience = 6).
* **FAISS Index** used for nearest-neighbor retrieval.
* **Local Mixup Module** replaces fixed interpolation with dynamic Beta-distributed λ.
* **Evaluation frameworks:**

  * `torchattacks` for reproducible ε-based attacks.
  * `foolbox` for normalized L₂-based PGD evaluation (Scenario 2).

---

## 🔬 10. Research Takeaways

1. **Retrieval guidance** significantly improves robustness under unseen perturbations.
2. **Adaptive local mixup** maintains accuracy without over-smoothing decision boundaries.
3. **Increasing K** (number of neighbors) yields diminishing returns but stabilizes defense.
4. Under white-box access (Scenario 1), RaCNN retains > 45 % accuracy under PGD,
   while baseline CNN collapses < 10 %.

---

## 🧠 Authors & Notes

* **Primary Implementation:** *[Your Name / Afra Anika]*
* **Collaborators:** Research team (adversarial defense & retrieval studies)
* **Frameworks:** PyTorch | TorchAttacks | Foolbox | FAISS | Matplotlib
* **Dataset:** CIFAR-10

> This work reproduces core results of *Zhao & Cho (ICLR 2019)* and extends them with
> adaptive mixup and normalized L₂ Foolbox benchmarking for comprehensive robustness evaluation.

---

## 🖼️ Example Output (Scenario 1)

<p align="center">
  <img src="artifacts/scenario1_attack_curves_final.png" width="80%">
</p>

## 🖼️ Example Output (Scenario 2)

<p align="center">
  <img src="artifacts/scenario2_foolbox_curves.png" width="80%">
</p>

---

## 📜 Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{zhao2019racnn,
  title={Retrieval-Augmented Convolutional Neural Networks for Improved Robustness against Adversarial Examples},
  author={Jake Zhao and Kyunghyun Cho},
  booktitle={ICLR},
  year={2019}
}
```

---

**🧩 Status:**
✅ Training complete (Baseline, RaCNN K = 5 & 10)
✅ Scenario 1 (TorchAttacks) finished
✅ Scenario 2 (Foolbox FGSM/iFGSM for K10) finished
