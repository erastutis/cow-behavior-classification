from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml


LABEL_COLS = ["y_stand", "y_lying_down", "y_foraging", "y_drinking_water", "y_rumination"]
LABEL_NAMES = ["stand", "lying_down", "foraging", "drinking_water", "rumination"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
