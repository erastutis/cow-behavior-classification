# Thesis-code consistency notes

This repository version is aligned with the bachelor thesis text.

## Key consistency points

- Dataset split: 70/15/15 by `video_id`.
- Multi-label formulation: five independent sigmoid outputs.
- ResNet-18: single-frame baseline, ImageNet pretrained, two-stage training.
- VideoViT: 8-frame clips, ViT-Base/16 ImageNet backbone, temporal multi-head attention.
- R3D-18 v1: 8-frame clips, Kinetics-400 pretrained R3D-18.
- R3D-18 ultra: 16-frame clips, Kinetics-400 pretrained R3D-18, modified classification head.
- Loss functions and sampling follow the thesis method section.
- Reported metric tables are included in `src/eval_all_models.py`.

## Important limitation

This repository excludes dataset files and model checkpoints. The scripts document the workflow and expected implementation logic. To reproduce the exact numerical results, the same dataset files, preprocessing, checkpoints and split must be available locally.
