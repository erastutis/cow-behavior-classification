from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models.video as video_models
from tqdm import tqdm

from _utils import LABELS, get_device, load_config, parse_config_arg, set_seed


class ClipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, clip_size: int = 8, image_size: int = 224):
        self.samples = []
        self.clip_size = clip_size
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        for _, grp in df.groupby("video_id"):
            grp = grp.sort_values("frame_path").reset_index(drop=True)
            step = max(1, clip_size // 2)
            for start in range(0, len(grp) - clip_size + 1, step):
                chunk = grp.iloc[start:start + clip_size]
                labels = chunk[LABELS].values.astype("float32").max(axis=0)
                self.samples.append((chunk["frame_path"].tolist(), labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        paths, labels = self.samples[idx]
        frames = [self.tf(Image.open(p).convert("RGB")) for p in paths]
        x = torch.stack(frames, dim=1)  # C, T, H, W
        y = torch.tensor(labels)
        return x, y


def build_model(num_classes: int = 5) -> nn.Module:
    model = video_models.r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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

    model = build_model(num_classes=len(LABELS)).to(device)
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

    torch.save(model.state_dict(), out_dir / "r3d18_best.pth")
    print(f"Saved model to {out_dir / 'r3d18_best.pth'}")


if __name__ == "__main__":
    main()
