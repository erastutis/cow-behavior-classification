from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

from _utils import LABEL_NAMES


def build_resnet18(weights_path: Path, device: str):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_NAMES))
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_resnet18(Path(args.weights), device)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image).convert("RGB")
    x = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].cpu().numpy()

    predictions = {
        name: float(prob)
        for name, prob in zip(LABEL_NAMES, probs)
        if prob >= args.threshold
    }

    print("Predictions:")
    for k, v in predictions.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()
