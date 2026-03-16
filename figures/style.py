"""
Nature Communications figure style helpers.

Provides rcParams, sizing constants, colourblind-safe palettes,
and dual-format (PDF + TIFF) export with LZW compression.

This module is intentionally side-effect free on import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml

# ---------------------------------------------------------------------------
# Nature Communications dimension constants (mm)
# ---------------------------------------------------------------------------
SINGLE_COL_MM = 89
DOUBLE_COL_MM = 183
MAX_HEIGHT_MM = 170

# ---------------------------------------------------------------------------
# Colourblind-friendly palette (Wong 2011, Viridis-anchored)
# ---------------------------------------------------------------------------
PALETTE_WONG = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
]

PALETTE_VIRIDIS_8 = [
    "#440154",
    "#46327e",
    "#365c8d",
    "#277f8e",
    "#1fa187",
    "#4ac16d",
    "#9FDA3A",
    "#FDE725",
]

# Semantic role palette — consistent colors across restructured figures
SEMANTIC_PALETTE = {
    "physics": "#0072B2",    # blue — physical/H-based quantities
    "learned": "#009E73",    # green — learned/AI quantities
    "ablation": "#D55E00",   # vermilion — ablation/baseline comparisons
    "highlight": "#E69F00",  # orange — highlighted ablation variant
}


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
def mm_to_in(mm: float) -> float:
    """Convert millimetres to inches."""
    return mm / 25.4


def in_to_mm(inches: float) -> float:
    """Convert inches to millimetres."""
    return inches * 25.4


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
def set_nature_rcparams(base_fontsize: int = 8) -> None:
    """Apply Nature Communications–compliant Matplotlib rcParams.

    Fonts: sans-serif (Arial > Helvetica > DejaVu Sans).
    All fonts embedded as TrueType (Type 42) in PDF/PS.
    """
    mpl.rcParams.update(
        {
            # Fonts — sans-serif ONLY (Nature requirement).
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
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
            # Embed fonts in PDF/PS — Type 42 (TrueType).
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------------
# Figure creation
# ---------------------------------------------------------------------------
def make_figure(width_mm: float = SINGLE_COL_MM, height_mm: float = 60) -> plt.Figure:
    """Create a figure with dimensions in mm."""
    return plt.figure(figsize=(mm_to_in(width_mm), mm_to_in(height_mm)))


def add_panel_label(
    ax: mpl.axes.Axes,
    label: str,
    x: float = 0.0,
    y: float = 1.02,
) -> None:
    """Add a bold, lowercase panel label (a, b, c …) in Nature style."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


# ---------------------------------------------------------------------------
# Dual-format save (PDF + TIFF)
# ---------------------------------------------------------------------------
def save_outputs(
    fig: plt.Figure,
    out_prefix: str | Path,
    dpi_tiff: int = 300,
) -> list[Path]:
    """Save figure as PDF (vector) and TIFF (LZW-compressed raster).

    Saves with exact figsize dimensions (no ``bbox_inches="tight"``),
    which is required for Nature Communications width compliance.

    Returns list of output paths [pdf_path, tiff_path].
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = out_prefix.with_suffix(".pdf")
    tiff_path = out_prefix.with_suffix(".tiff")

    fig.savefig(str(pdf_path))
    fig.savefig(
        str(tiff_path),
        dpi=dpi_tiff,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    _write_layout_metadata(fig, out_prefix)
    return [pdf_path, tiff_path]


def collect_layout_metadata(fig: plt.Figure) -> dict[str, Any]:
    """Collect figure and axes geometry in mm for downstream review tooling."""
    width_in, height_in = fig.get_size_inches()
    width_mm = in_to_mm(width_in)
    height_mm = in_to_mm(height_in)

    axes_payload: list[dict[str, Any]] = []
    for idx, ax in enumerate(fig.axes):
        bbox = ax.get_position()
        x_mm = bbox.x0 * width_mm
        y_mm = bbox.y0 * height_mm
        w_mm = bbox.width * width_mm
        h_mm = bbox.height * height_mm
        title = ax.get_title()
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        axes_payload.append(
            {
                "index": idx,
                "kind": getattr(ax, "name", "axes"),
                "has_data": bool(ax.has_data()),
                "title": title or None,
                "xlabel": xlabel or None,
                "ylabel": ylabel or None,
                "bbox_norm": {
                    "x0": round(float(bbox.x0), 6),
                    "y0": round(float(bbox.y0), 6),
                    "width": round(float(bbox.width), 6),
                    "height": round(float(bbox.height), 6),
                },
                "bbox_mm": {
                    "x0": round(float(x_mm), 3),
                    "y0": round(float(y_mm), 3),
                    "width": round(float(w_mm), 3),
                    "height": round(float(h_mm), 3),
                },
            }
        )

    return {
        "figure_mm": {
            "width": round(float(width_mm), 3),
            "height": round(float(height_mm), 3),
        },
        "axes": axes_payload,
    }


def _write_layout_metadata(fig: plt.Figure, out_prefix: str | Path) -> Path:
    """Persist a sidecar JSON file with figure/axes layout metadata."""
    out_prefix = Path(out_prefix)
    layout_path = out_prefix.with_suffix(".layout.json")
    payload = collect_layout_metadata(fig)
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return layout_path


# ---------------------------------------------------------------------------
# Path configuration loader
# ---------------------------------------------------------------------------
def load_paths(yaml_path: str | Path | None = None) -> dict[str, Any]:
    """Load data paths from conf/paths.yaml.

    If *yaml_path* is ``None``, defaults to ``figures/conf/paths.yaml``
    relative to the repository root (two levels up from this file).
    """
    if yaml_path is None:
        yaml_path = Path(__file__).resolve().parent / "conf" / "paths.yaml"
    else:
        yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Path config not found: {yaml_path}")

    with open(yaml_path) as f:
        return yaml.safe_load(f)
