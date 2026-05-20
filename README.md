# Cattle Behavior Classification Using Deep Neural Networks

This repository contains the code structure and reproducibility files for the bachelor thesis:

**Application of Deep Neural Networks for Cattle Behavior Classification from Video Data**  
Author: **Eimantas Raštutis**  
Institution: **Kaunas University of Technology, Faculty of Electrical and Electronics Engineering**

## Project description

The project investigates **multi-label cattle behavior classification** from image and video data using deep learning models.

The task uses five behavior labels:

- `stand`
- `lying_down`
- `foraging`
- `drinking_water`
- `rumination`

The experiments compare four model configurations:

- `ResNet-18`
- `VideoViT`
- `R3D-18 v1`
- `R3D-18 ultra`

## Dataset

The dataset is **not included** in this repository due to size and licensing reasons.

Expected local structure:

```text
data/
└── CBVD-5/
    ├── labelframes/
    └── annotations.csv
```

The code expects a processed manifest file with the following columns:

```text
frame_path, video_id, y_stand, y_lying_down, y_foraging, y_drinking_water, y_rumination, split
```

## Main workflow

```bash
python src/prepare_dataset.py --config configs/resnet18.yaml
python src/train_resnet18.py --config configs/resnet18.yaml
python src/train_videovit.py --config configs/videovit.yaml
python src/train_r3d18.py --config configs/r3d18_v1.yaml
python src/train_r3d18_ultra.py --config configs/r3d18_ultra.yaml
python src/eval_all_models.py --config configs/eval.yaml
```

## Main reported results

| Model | Macro F1 | Micro F1 | mAP | Clip length | Parameters |
|---|---:|---:|---:|---:|---:|
| ResNet-18 | 0.510 | 0.594 | - | 1 | ~11M |
| VideoViT | 0.643 | 0.777 | - | 8 | ~87M |
| R3D-18 v1 | 0.734 | 0.817 | 0.757 | 8 | ~33M |
| R3D-18 ultra | 0.773 | 0.843 | 0.773 | 16 | ~34M |

## Per-class F1 comparison

| Class | ResNet-18 | VideoViT | R3D-18 v1 | R3D-18 ultra |
|---|---:|---:|---:|---:|
| stand | 0.800 | 0.889 | 0.918 | 0.925 |
| lying_down | 0.640 | 0.875 | 0.890 | 0.919 |
| foraging | 0.500 | 0.616 | 0.814 | 0.836 |
| rumination | 0.500 | 0.728 | 0.711 | 0.755 |
| drinking_water | 0.100 | 0.103 | 0.337 | 0.429 |

## Notes

Large files are intentionally excluded from GitHub:

- dataset files
- raw frames
- videos
- trained weights
- temporary checkpoints

The repository documents the experimental pipeline and implementation logic. Exact results depend on the original Colab runs, saved checkpoints and validation-based model selection.
