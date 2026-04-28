# Face Anti-Spoofing Classifier

A production-ready deep learning system for detecting face presentation attacks. The pipeline classifies facial images into six categories, including a real person and five types of spoof attacks, using a CLIP-pretrained Vision Transformer with a fully automated, self-correcting training strategy.



---

## Project Overview
The emphasis of this project is on **data quality** and **training robustness**: rather than relying solely on model capacity, the pipeline actively detects and corrects label noise, handles real-world image inconsistencies, and augments training data through pseudolabeling.

| **Objective** | Information |
|---|---|
| **Task** | Multi-class face presentation attack detection |
| **Classes** | 6 (1 real, 5 spoof types) |
| **Model** | ViT-Base CLIP (`vit_base_patch16_clip_224.openai_ft_in12k_in1k`) |
| **Metric** | Macro F1-Score |
| **Training** | 2-round self-correcting pipeline with 5-fold cross-validation |

---

## Detected Attack Types

| Class | Description |
|---|---|
| `realperson` | Genuine live face |
| `fake_mannequin` | 3D mannequin or doll |
| `fake_mask` | Physical face mask |
| `fake_printed` | Printed photo attack |
| `fake_screen` | Digital screen replay |
| `fake_unknown` | Other / unclassified spoof |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-ViT--Base-blueviolet?style=flat)
![Albumentations](https://img.shields.io/badge/Albumentations-FF6F00?style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)

| Category | Tools |
|---|---|
| **Deep Learning** | PyTorch, timm, AMP (mixed precision) |
| **Backbone** | ViT-Base CLIP pretrained on 400M images |
| **Augmentation** | Albumentations, custom TTA |
| **Data Processing** | OpenCV, Pillow, Pandas, NumPy |
| **Evaluation** | scikit-learn (F1, classification report, confusion matrix) |
| **Optimization** | SciPy (Nelder-Mead threshold optimizer) |
| **Environment** | Kaggle GPU (T4/P100) |

---

## Pipeline

```mermaid
flowchart TD
    A([Raw Image Dataset]) --> B[Data Cleaning] --> C[Round 1 Training] --> D[Evaluate OOF] --> F[Auto-Relabeling + Pseudolabeling]
    F --> G[Round 2 Training with relabeled + pseudo data]
    G --> H[Final Evaluation with TTA × 6 variants]
```

---

## Technical Highlights

### Self-Correcting Label Noise Detection

One of the core contributions of this pipeline is its ability to identify and correct mislabeled training samples fully automatically. After Round 1 training, out-of-fold (OOF) predictions are analyzed to flag images where the model is highly confident in a *different* class than the assigned label.
**Relabeling criteria (deterministic, no human intervention):**
- Model confidence toward the predicted class ≥ 90%
- Model confidence toward the original label ≤ 5%

### Pseudolabeling for Unlabeled Data
Unlabeled images with high-confidence predictions (≥ 85%) are added to the training set before Round 2, effectively expanding the training distribution and improving generalization on harder examples.

### Two-Phase Training Strategy
| Phase | Epochs | LR | Backbone |
|---|---|---|---|
| Phase 1 | 3 | 5e-4 | Frozen (head only) |
| Phase 2 | 15 | 2e-5 | Unfrozen (full fine-tune) |

Phase 1 prevents gradient instability by warming up the classification head before the backbone is exposed to large gradient updates. This is particularly important for small datasets.

### Augmentation Pipeline
Training augmentations are designed to simulate real-world spoof artifacts:
- **Blur**: motion blur & Gaussian blur (simulate screen/print artifacts)
- **Compression**: JPEG quality 40–95 (simulate re-encoded or re-photographed images)
- **Color/lighting**: brightness, contrast, color jitter, hue/saturation, gamma
- **Noise**: Gaussian noise & ISO noise (simulate sensor variability)
- **Geometry**: random crop, flip, shift, scale, rotation, shadow
- **Occlusion**: CoarseDropout (random patches masked out)

**TTA (Test-Time Augmentation):** 6 variants averaged at inference, include original, horizontal flip, brightness+, brightness−, slight rotation, center crop from slightly larger resize.

---

## Model Architecture

```mermaid
flowchart TD
    A["Input Image (224×224)"] --> B["ViT-Base CLIP Backbone (pretrained on 400M images)"]
    B --> C["Feature dim: 768 — LayerNorm → Dropout(0.25)"]
    C --> D["Linear(768 → 6)"]
    D --> E["Class Logits (6 classes)"]
```

- **Backbone:** `vit_base_patch16_clip_224.openai_ft_in12k_in1k` via `timm`
- **Loss:** Weighted Label Smoothing Cross-Entropy — handles class imbalance via inverse-frequency class weights, smoothing factor 0.1
- **Optimizer:** AdamW with weight decay 1e-4
- **Scheduler:** Linear warmup → Cosine annealing (Phase 1) / Cosine annealing (Phase 2)
- **Regularization:** Mixup (α=0.2), label smoothing, dropout, drop path (0.1)
- **Mixed precision:** AMP on GPU with compute capability ≥ 7.0

---

## Data Quality Pipeline

Real-world face anti-spoofing datasets are noisy. This pipeline addresses the following issues systematically:

| Issue | Solution |
|---|---|
| Duplicate images across classes (label noise) | Perceptual hash (MD5 of 16×16 grayscale) to remove all copies |
| Duplicate images within same class | Perceptual hash to keep one copy |
| Tilted/rotated images from smartphones | Auto-rotate using PIL EXIF tag 274 before augmentation |
| Mislabeled images | OOF-based auto-relabeling with deterministic confidence thresholds |
| Resolution mismatch (256px–4096px) | Normalize max side to 1024px to prevent blur on small images |
| Extremely dark/corrupt images | Brightness threshold — remove images with mean pixel value < 20 |
| Class imbalance | Inverse-frequency class weights in loss function |

---

## Results

<p align="center">
  <img src="assets/confusion_matrix.png" width="45%" alt="Confusion Matrix"/>
  <img src="assets/relabel.png" width="45%" alt="Auto-Relabeling Candidates"/>
</p>
<p align="center">
  <img src="assets/round_2_predict.png" width="60%" alt="Round 2 Sample Predictions"/>
</p>

---

## Reproducibility

The entire pipeline is deterministic and requires zero manual steps:

- `set_seed(42)` locks NumPy, PyTorch, CUDA, and Python hash randomness
- Auto-relabeling uses fixed confidence thresholds (`conf_new ≥ 0.90`, `conf_old ≤ 0.05`)
- Pseudolabeling uses a fixed threshold (`0.85`)

---

## Repository Structure

```
face-anti-spoofing/
│
├── README.md
│
├── notebook/
│   └── face_anti_spoofing.ipynb       # End-to-end notebook
│
├── assets/
│   ├── confusion_matrix.png           # Confusion matrix visualization
│   ├── relabel.png                    # Auto-relabeling candidates grid
│   └── round_2_predict.png            # Round 2 sample predictions
│
└── requirements.txt
```

---

## License

This project is built for portfolio and learning purposes. The dataset comes from Kaggle DAC FindIt 2026.

---

## Author

**Yan Andhinaya Ardika**
- Github: [@yandik](https://github.com/yanardika) 
