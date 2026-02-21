"""
Matplotlib helpers for Nature-style figures.

This module is intentionally side-effect free on import.
Example usage is guarded by `if __name__ == "__main__":`.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

def mm_to_in(mm):
    return mm / 25.4

def set_nature_rcparams(base_fontsize=8):
    mpl.rcParams.update({
        # Fonts (sans-serif; prefer Arial/Helvetica).
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

        # Font sizes (pt).
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,

        # Line widths (pt).
        "axes.linewidth": 0.8,
        "lines.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        # Clean white background.
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        # Embed fonts in PDF/PS to avoid substitution.
        "pdf.fonttype": 42,  # TrueType
        "ps.fonttype": 42,
    })

def make_figure(width_mm=88, height_mm=60):
    fig = plt.figure(figsize=(mm_to_in(width_mm), mm_to_in(height_mm)))
    return fig

def add_panel_label(ax, label, x=0.0, y=1.02):
    # Nature-style panel labels: bold lowercase a, b, c...
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="left")

def save_outputs(fig, out_prefix, dpi_tiff=300):
    # 1) Vector: PDF (preferred for line plots / statistical figures).
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")

    # 2) Raster: TIFF (sometimes required by submission systems).
    fig.savefig(f"{out_prefix}.tiff", dpi=dpi_tiff, bbox_inches="tight")

if __name__ == "__main__":
    # Demo (writes outputs under results/ to keep repo root clean).
    from pathlib import Path

    import numpy as np

    set_nature_rcparams(base_fontsize=8)

    out_dir = Path("results") / "nature_style_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = make_figure(width_mm=88, height_mm=60)
    ax = fig.add_subplot(1, 1, 1)

    x = np.linspace(0, 10, 200)
    ax.plot(x, np.sin(x), label="sin(x)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.legend(frameon=False)
    add_panel_label(ax, "a")

    save_outputs(fig, str(out_dir / "Fig1a_single_column"), dpi_tiff=300)
    plt.close(fig)
