# Cattle Behavior Classification Using Deep Neural Networks

This repository contains the code structure and reproducibility files for the bachelor thesis:

**Application of Deep Neural Networks for Cattle Behavior Classification from Video Data**  
Author: **Eimantas Raštutis**  
Faculty: **Kaunas University of Technology, Faculty of Electrical and Electronics Engineering**

## Project description

The project investigates cattle behavior classification from image and video data using deep learning models.  
The task is formulated as **multi-label classification**, because one frame can contain more than one active behavior label.

The analyzed behavior classes are:

- standing
- lying down
- foraging
- drinking water
- rumination

The compared model families are:

- ResNet-18
- R3D-18
- VideoViT

## Repository structure

```text
cow-behavior-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── src/
│   ├── prepare_dataset.py
│   ├── train_resnet18.py
│   ├── train_r3d18.py
│   ├── train_videovit.py
│   ├── eval_all_models.py
│   └── demo_inference.py
│
├── configs/
│   ├── resnet18.yaml
│   ├── r3d18.yaml
│   └── videovit.yaml
│
├── notebooks/
│   └── original_colab_experiments.ipynb
│
├── outputs/
│   └── figures/
│       └── confusion_matrices/
│
└── docs/
    └── thesis_results_summary.md
```

## Dataset

The dataset is **not included** in this repository due to size and licensing reasons.

Experiments were performed using the **CBVD-5** cattle behavior dataset.

Expected local structure:

```text
data/
└── CBVD-5/
    ├── labelframes/
    └── annotations.csv
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/cow-behavior-classification.git
cd cow-behavior-classification

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main workflow

### 1. Prepare dataset

```bash
python src/prepare_dataset.py --config configs/resnet18.yaml
```

### 2. Train frame-based ResNet-18 model

```bash
python src/train_resnet18.py --config configs/resnet18.yaml
```

### 3. Train R3D-18 video model

```bash
python src/train_r3d18.py --config configs/r3d18.yaml
```

### 4. Train VideoViT model

```bash
python src/train_videovit.py --config configs/videovit.yaml
```

### 5. Evaluate all models

```bash
python src/eval_all_models.py
```

## Main reported results

| Model | Macro F1 | Micro F1 |
|---|---:|---:|
| ResNet-18 | 0.510 | - |
| VideoViT | 0.643 | - |
| R3D-18 v1 | 0.734 | - |
| R3D-18 ultra | 0.773 | 0.843 |

## Notes

Large files are intentionally excluded from GitHub:

- dataset files
- trained model weights
- raw video/image data
- temporary training outputs

This repository is intended to document the experimental workflow and make the project easier to reproduce.
