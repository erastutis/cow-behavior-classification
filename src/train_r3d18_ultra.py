from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.models.video as video_models
from tqdm import tqdm

from _utils import LABEL_COLS, device, ensure_dir, load_config, parse_args, set_seed
from datasets import ClipDataset
from losses import CombinedLoss, compute_pos_weight


class R3D18Ultra(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.backbone = video_models.r3d_18(weights=video_models.R3D_18_Weights.KINETICS400_V1)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.fc(self.backbone(x))


def make_sampler(dataset: ClipDataset) -> WeightedRandomSampler:
    labels = torch.stack([torch.tensor(s[1]) for s in dataset.samples])
    class_freq = labels.sum(dim=0).clamp(min=1)
    class_weight = 1.0 / class_freq
    sample_weight = (labels * class_weight).sum(dim=1)
    sample_weight = torch.where(sample_weight > 0, sample_weight, torch.ones_like(sample_weight) * sample_weight.mean())
    return WeightedRandomSampler(sample_weight.double(), num_samples=len(sample_weight), replacement=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    dev = device()

    df = pd.read_csv(cfg["data"]["manifest_path"])
    train_df = df[df["split"] == "train"].copy()

    train_ds = ClipDataset(
        train_df,
        clip_size=cfg["model"]["clip_size"],
        clip_stride=cfg["model"]["clip_stride"],
        image_size=cfg["model"]["image_size"],
        train=True,
    )

    sampler = make_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 2),
    )

    labels = torch.tensor(train_df[LABEL_COLS].values.astype("float32"))
    pos_weight = compute_pos_weight(labels, max_weight=cfg["training"].get("max_pos_weight", 8.0)).to(dev)

    model = R3D18Ultra(num_classes=len(LABEL_COLS)).to(dev)
    criterion = CombinedLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg["training"]["backbone_lr"]},
        {"params": model.fc.parameters(), "lr": cfg["training"]["head_lr"]},
    ], weight_decay=cfg["training"].get("weight_decay", 0.0001))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    best_loss = float("inf")
    patience = cfg["training"]["early_stopping_patience"]
    bad_epochs = 0

    out_dir = ensure_dir(cfg["training"]["output_dir"])

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total = 0.0

        for x, y in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)

        scheduler.step()
        epoch_loss = total / len(train_ds)
        print(f"epoch {epoch + 1}: loss={epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir / "r3d18_ultra_best.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    print(f"Saved: {out_dir / 'r3d18_ultra_best.pth'}")


if __name__ == "__main__":
    main()
