from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml


LABELS = ["y_stand", "y_lying_down", "y_foraging", "y_drinking_water", "y_rumination"]
LABEL_NAMES = ["stand", "lying_down", "foraging", "drinking_water", "rumination"]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_config_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()
