from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from _utils import LABELS, get_device, load_config, parse_config_arg, set_seed


class FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["frame_path"]).convert("RGB")
        x = self.tf(image)
        y = torch.tensor(row[LABELS].values.astype("float32"))
        return x, y


def build_model(num_classes: int = 5) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="train"):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def main() -> None:
    args = parse_config_arg()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = get_device()
    manifest = Path(cfg["data"]["manifest_path"])
    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)
    train_df = df[df["split"] == "train"].copy()

    dataset = FrameDataset(train_df, image_size=cfg["model"]["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 2),
    )

    model = build_model(num_classes=len(LABELS)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    for epoch in range(cfg["training"]["epochs"]):
        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}: train_loss={loss:.4f}")

    torch.save(model.state_dict(), out_dir / "resnet18_best.pth")
    print(f"Saved model to {out_dir / 'resnet18_best.pth'}")


if __name__ == "__main__":
    main()
