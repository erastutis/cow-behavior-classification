from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _utils import LABEL_COLS, load_config, parse_args, set_seed


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    annotations_path = Path(cfg["data"]["annotations_path"])
    manifest_path = Path(cfg["data"]["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(annotations_path)

    missing = [c for c in ["frame_path", "video_id", *LABEL_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    videos = df["video_id"].dropna().unique()
    rng = np.random.default_rng(cfg.get("seed", 42))
    rng.shuffle(videos)

    n = len(videos)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_videos = videos[:n_train]
    val_videos = videos[n_train:n_train + n_val]
    test_videos = videos[n_train + n_val:]

    df["split"] = "train"
    df.loc[df["video_id"].isin(val_videos), "split"] = "val"
    df.loc[df["video_id"].isin(test_videos), "split"] = "test"

    df.to_csv(manifest_path, index=False)

    print(f"Saved manifest: {manifest_path}")
    print(df["split"].value_counts())
    print(df[LABEL_COLS].sum().astype(int))


if __name__ == "__main__":
    main()
