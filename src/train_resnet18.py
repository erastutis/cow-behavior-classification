from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm

from _utils import LABEL_COLS, device, ensure_dir, load_config, parse_args, set_seed
from datasets import FrameDataset
from losses import FocalLoss, compute_pos_weight


def make_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    labels = torch.tensor(df[LABEL_COLS].values.astype("float32"))
    class_freq = labels.sum(dim=0).clamp(min=1)
    class_weight = 1.0 / class_freq
    sample_weight = (labels * class_weight).sum(dim=1)
    sample_weight = torch.where(sample_weight > 0, sample_weight, torch.ones_like(sample_weight) * sample_weight.mean())
    return WeightedRandomSampler(sample_weight.double(), num_samples=len(sample_weight), replacement=True)


def build_model(num_classes: int = 5):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, dev: str):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, desc="train"):
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    dev = device()

    df = pd.read_csv(cfg["data"]["manifest_path"])
    train_df = df[df["split"] == "train"].copy()

    train_ds = FrameDataset(train_df, image_size=cfg["model"]["image_size"], train=True)
    sampler = make_sampler(train_df) if cfg["training"].get("weighted_sampler", False) else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg["training"].get("num_workers", 2),
    )

    labels = torch.tensor(train_df[LABEL_COLS].values.astype("float32"))
    pos_weight = compute_pos_weight(labels).to(dev)

    model = build_model(num_classes=len(LABEL_COLS)).to(dev)

    # Stage 1: freeze backbone, train classification head.
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")

    criterion = FocalLoss(gamma=cfg["training"]["focal_gamma"], pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=cfg["training"]["head_lr"])

    for epoch in range(cfg["training"]["head_epochs"]):
        loss = run_epoch(model, train_loader, criterion, optimizer, dev)
        print(f"head epoch {epoch + 1}: loss={loss:.4f}")

    # Stage 2: unfreeze and fine-tune full model.
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["finetune_lr"])
    for epoch in range(cfg["training"]["finetune_epochs"]):
        loss = run_epoch(model, train_loader, criterion, optimizer, dev)
        print(f"finetune epoch {epoch + 1}: loss={loss:.4f}")

    out_dir = ensure_dir(cfg["training"]["output_dir"])
    torch.save(model.state_dict(), out_dir / "resnet18_best.pth")
    print(f"Saved: {out_dir / 'resnet18_best.pth'}")


if __name__ == "__main__":
    main()
