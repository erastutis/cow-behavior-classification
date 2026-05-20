from __future__ import annotations

import torch
import torch.nn as nn
import timm

from train_r3d18 import ClipDataset
from _utils import LABELS, get_device, load_config, parse_config_arg, set_seed

from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        pooled = feats.mean(dim=1)
        return self.head(pooled)


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

    dataset = ClipDataset(
        train_df,
        clip_size=cfg["model"]["clip_size"],
        image_size=cfg["model"]["image_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 2),
    )

    model = VideoViT(num_classes=len(LABELS)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(loader, desc=f"epoch {epoch + 1}"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        print(f"Epoch {epoch + 1}: train_loss={total_loss / len(dataset):.4f}")

    torch.save(model.state_dict(), out_dir / "videovit_best.pth")
    print(f"Saved model to {out_dir / 'videovit_best.pth'}")


if __name__ == "__main__":
    main()
