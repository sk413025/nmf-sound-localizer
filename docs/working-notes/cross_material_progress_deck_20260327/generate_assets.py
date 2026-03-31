from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


DECK_DIR = Path(__file__).resolve().parent
ASSET_DIR = DECK_DIR / "assets"
OUTPUT = ASSET_DIR / "workflow_overview.png"


def add_box(ax, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5,
        edgecolor="#263238",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#102027",
    )


def add_arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#455a64"),
    )


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 4.3), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.02,
        0.93,
        "Cross-material white-noise progress workflow",
        fontsize=18,
        fontweight="bold",
        color="#102027",
    )
    ax.text(
        0.02,
        0.865,
        "Input: 5 materials x 37 angles = 185 recordings from cross-materials.zip",
        fontsize=11,
        color="#455a64",
    )

    y = 0.38
    w = 0.16
    h = 0.23
    xs = [0.02, 0.215, 0.41, 0.605, 0.80]
    labels = [
        "1. Audit ZIP\ninventory + protocol",
        "2. Legacy ingest\nwhite_noise roots",
        "3. Reproduce H\nOriginal -> Material",
        "4. Screen materials\nTop-1 / MAE",
        "5. Fig. 6 support\nrank + interpretation",
    ]
    colors = ["#d9f0ff", "#e3f2dc", "#fff3cd", "#fde2e4", "#ede7f6"]

    for x, label, color in zip(xs, labels, colors):
        add_box(ax, x, y, w, h, label, color)

    for idx in range(len(xs) - 1):
        add_arrow(ax, xs[idx] + w, y + h / 2, xs[idx + 1], y + h / 2)

    ax.text(
        0.02,
        0.15,
        "Deck scope: executed white-noise chain only. Speech results are intentionally excluded.",
        fontsize=11,
        color="#37474f",
    )
    ax.text(
        0.02,
        0.09,
        "Main decision point: B is primary, W is backup; stronger coherence alone does not guarantee better downstream screening.",
        fontsize=11,
        color="#37474f",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
