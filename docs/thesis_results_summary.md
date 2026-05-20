# Thesis results summary

This document summarizes the main experimental setup and reported results.

## Task

Multi-label cattle behavior classification from image/video data.

## Classes

1. standing
2. lying down
3. foraging
4. drinking water
5. rumination

## Compared models

- ResNet-18
- VideoViT
- R3D-18 v1
- R3D-18 ultra

## Main metrics

- Macro F1
- Micro F1
- per-class F1
- Average Precision
- multi-label confusion matrices

## Reported model comparison

| Model | Macro F1 | Micro F1 |
|---|---:|---:|
| ResNet-18 | 0.510 | - |
| VideoViT | 0.643 | - |
| R3D-18 v1 | 0.734 | - |
| R3D-18 ultra | 0.773 | 0.843 |

## Practical interpretation

The frame-based ResNet-18 model is useful as a baseline, but it does not directly model temporal information.  
R3D-18 performs better because it processes short video clips and can capture motion-related behavior patterns.  
The weakest class is drinking water because it has the fewest examples and is visually more dependent on context.
