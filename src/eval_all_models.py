from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_NAMES = ["stand", "lying_down", "foraging", "drinking_water", "rumination"]


REPORTED_RESULTS = pd.DataFrame([
    {
        "Model": "ResNet-18",
        "Macro F1": 0.510,
        "Micro F1": 0.594,
        "mAP": np.nan,
        "Clip length": 1,
        "Parameters": "~11M",
    },
    {
        "Model": "VideoViT",
        "Macro F1": 0.643,
        "Micro F1": 0.777,
        "mAP": np.nan,
        "Clip length": 8,
        "Parameters": "~87M",
    },
    {
        "Model": "R3D-18 v1",
        "Macro F1": 0.734,
        "Micro F1": 0.817,
        "mAP": 0.757,
        "Clip length": 8,
        "Parameters": "~33M",
    },
    {
        "Model": "R3D-18 ultra",
        "Macro F1": 0.773,
        "Micro F1": 0.843,
        "mAP": 0.773,
        "Clip length": 16,
        "Parameters": "~34M",
    },
])


PER_CLASS_F1 = pd.DataFrame({
    "Class": LABEL_NAMES,
    "ResNet-18": [0.800, 0.640, 0.500, 0.100, 0.500],
    "VideoViT": [0.889, 0.875, 0.616, 0.103, 0.728],
    "R3D-18 v1": [0.918, 0.890, 0.814, 0.337, 0.711],
    "R3D-18 ultra": [0.925, 0.919, 0.836, 0.429, 0.755],
})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=False,
        type=str,
        default="configs/eval.yaml",
        help="Palikta suderinamumui. Šiame skripte config nebūtinas.",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        type=str,
        default="outputs/figures",
        help="Kur išsaugoti sugeneruotus grafikus.",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_lt(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def add_bar_labels(ax, bars, decimals: int = 3, fontsize: int = 9, y_offset: int = 4) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            format_lt(float(height), decimals),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def apply_academic_style(ax) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=10)


def plot_macro_micro(out_dir: Path) -> None:
    x = np.arange(len(REPORTED_RESULTS))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_macro = ax.bar(
        x - width / 2,
        REPORTED_RESULTS["Macro F1"],
        width,
        label="Makro F1",
    )
    bars_micro = ax.bar(
        x + width / 2,
        REPORTED_RESULTS["Micro F1"],
        width,
        label="Mikro F1",
    )

    add_bar_labels(ax, bars_macro, decimals=3, fontsize=9)
    add_bar_labels(ax, bars_micro, decimals=3, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(REPORTED_RESULTS["Model"], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 reikšmė", fontsize=11)

    ax.legend(
        fontsize=9,
        loc="upper left",
        frameon=True,
    )

    apply_academic_style(ax)

    fig.tight_layout()
    fig.savefig(out_dir / "macro_micro_f1.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "macro_micro_f1.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_per_class(out_dir: Path) -> None:
    x = np.arange(len(PER_CLASS_F1["Class"]))
    models = ["ResNet-18", "VideoViT", "R3D-18 v1", "R3D-18 ultra"]
    width = 0.19

    fig, ax = plt.subplots(figsize=(11.5, 5.8))

    for i, model in enumerate(models):
        bars = ax.bar(
            x + (i - 1.5) * width,
            PER_CLASS_F1[model],
            width,
            label=model,
        )
        add_bar_labels(ax, bars, decimals=2, fontsize=8, y_offset=3)

    ax.set_xticks(x)
    ax.set_xticklabels(PER_CLASS_F1["Class"], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 reikšmė", fontsize=11)

    ax.legend(
        fontsize=9,
        loc="upper left",
        ncol=2,
        frameon=True,
    )

    apply_academic_style(ax)

    fig.tight_layout()
    fig.savefig(out_dir / "per_class_f1.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "per_class_f1.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    REPORTED_RESULTS.to_csv(out_dir / "reported_results.csv", index=False)
    PER_CLASS_F1.to_csv(out_dir / "per_class_f1.csv", index=False)

    plot_macro_micro(out_dir)
    plot_per_class(out_dir)

    print("Sugeneruota:")
    print(out_dir / "macro_micro_f1.png")
    print(out_dir / "macro_micro_f1.pdf")
    print(out_dir / "per_class_f1.png")
    print(out_dir / "per_class_f1.pdf")


if __name__ == "__main__":
    main()