from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from _utils import LABELS, load_config, parse_config_arg, set_seed


def main() -> None:
    args = parse_config_arg()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    annotations_path = Path(cfg["data"]["annotations_path"])
    output_manifest = Path(cfg["data"]["manifest_path"])
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotations_path}")

    df = pd.read_csv(annotations_path)

    missing = [c for c in LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns in annotation file: {missing}")

    if "video_id" not in df.columns:
        raise ValueError("Expected column 'video_id' for video-level train/val/test split.")

    videos = df["video_id"].dropna().unique()
    train_videos, temp_videos = train_test_split(
        videos,
        test_size=0.30,
        random_state=cfg.get("seed", 42),
    )
    val_videos, test_videos = train_test_split(
        temp_videos,
        test_size=0.50,
        random_state=cfg.get("seed", 42),
    )

    df["split"] = "train"
    df.loc[df["video_id"].isin(val_videos), "split"] = "val"
    df.loc[df["video_id"].isin(test_videos), "split"] = "test"

    df.to_csv(output_manifest, index=False)
    print(f"Saved manifest: {output_manifest}")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
