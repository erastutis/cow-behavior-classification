from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from _utils import LABEL_COLS


def build_transforms(image_size: int = 224, train: bool = False):
    ops = [transforms.Resize((image_size, image_size))]

    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ])

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(ops)


class FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 224, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = build_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["frame_path"]).convert("RGB")
        image = self.transform(image)
        labels = row[LABEL_COLS].values.astype("float32")
        labels = torch.tensor(labels)
        return image, labels


class ClipDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        clip_size: int = 8,
        clip_stride: int = 4,
        image_size: int = 224,
        train: bool = False,
    ):
        self.samples = []
        self.transform = build_transforms(image_size=image_size, train=train)

        for _, frames_df in df.groupby("video_id"):
            frames_df = frames_df.sort_values("frame_path").reset_index(drop=True)
            if len(frames_df) < clip_size:
                continue

            for start in range(0, len(frames_df) - clip_size + 1, clip_stride):
                clip_df = frames_df.iloc[start:start + clip_size]
                clip_paths = clip_df["frame_path"].tolist()
                clip_labels = clip_df[LABEL_COLS].max(axis=0).values.astype("float32")
                self.samples.append((clip_paths, clip_labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        paths, labels = self.samples[idx]
        frames = [self.transform(Image.open(path).convert("RGB")) for path in paths]
        clip = torch.stack(frames, dim=1)  # C, T, H, W
        labels = torch.tensor(labels, dtype=torch.float32)
        return clip, labels
