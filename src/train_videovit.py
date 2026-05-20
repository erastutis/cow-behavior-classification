from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

from _utils import LABEL_COLS, device, ensure_dir, load_config, parse_args, set_seed
from datasets import ClipDataset
from losses import compute_pos_weight


class VideoViT(nn.Module):
    def __init__(self, num_classes: int = 5, hidden: int = 768):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.temporal_attn = nn.MultiheadAttention(hidden, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        # x: B, C, T, H, W
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        feats = self.vit(x)
        feats = feats.reshape(b, t, -1)
        attn_out, _ = self.temporal_attn(feats, feats, feats)
        feats = self.norm(feats + attn_out)
        return self.head(feats.mean(dim=1))


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
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 2),
    )

    labels = torch.tensor(train_df[LABEL_COLS].values.astype("float32"))
    pos_weight = compute_pos_weight(labels).to(dev)

    model = VideoViT(num_classes=len(LABEL_COLS)).to(dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"].get("weight_decay", 0.0001))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

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
        print(f"epoch {epoch + 1}: loss={total / len(train_ds):.4f}")

    out_dir = ensure_dir(cfg["training"]["output_dir"])
    torch.save(model.state_dict(), out_dir / "videovit_best.pth")
    print(f"Saved: {out_dir / 'videovit_best.pth'}")


if __name__ == "__main__":
    main()
