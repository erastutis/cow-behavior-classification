from __future__ import annotations

# Minimal single-image demo for the frame-based ResNet-18 model.
# Video models require clips, so they are not included in this simple demo script.

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision import transforms

from _utils import LABEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_NAMES))
    model.load_state_dict(torch.load(args.weights, map_location=dev))
    model.to(dev).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(dev)

    with torch.no_grad():
        probs = torch.sigmoid(model(image))[0].cpu().numpy()

    for name, p in zip(LABEL_NAMES, probs):
        if p >= args.threshold:
            print(f"{name}: {p:.3f}")


if __name__ == "__main__":
    main()
