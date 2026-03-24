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
from matplotlib.transforms import Bbox
import yaml

from figures.layout_contract import (
    contract_version,
    font_pt,
    font_tokens,
    source_layout_spec,
)

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
    "classical": "#E69F00",  # orange — classical analytical references (OMP)
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
        fontsize=font_pt("panel_label"),
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
    *,
    typography: dict[str, float] | None = None,
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
    _write_layout_metadata(fig, out_prefix, typography=typography)
    return [pdf_path, tiff_path]


def collect_layout_metadata(
    fig: plt.Figure,
    *,
    typography: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Collect figure and axes geometry in mm for downstream review tooling."""
    width_in, height_in = fig.get_size_inches()
    width_mm = in_to_mm(width_in)
    height_mm = in_to_mm(height_in)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _figure_norm_payload(bbox: Bbox | None) -> dict[str, Any] | None:
        if bbox is None:
            return None
        x0 = float(bbox.x0)
        y0 = float(bbox.y0)
        w = float(bbox.width)
        h = float(bbox.height)
        return {
            "bbox_norm": {
                "x0": round(x0, 6),
                "y0": round(y0, 6),
                "width": round(w, 6),
                "height": round(h, 6),
            },
            "bbox_mm": {
                "x0": round(x0 * width_mm, 3),
                "y0": round(y0 * height_mm, 3),
                "width": round(w * width_mm, 3),
                "height": round(h * height_mm, 3),
            },
        }

    def _display_payload(bbox: Bbox | None) -> dict[str, Any] | None:
        if bbox is None:
            return None
        bbox_fig = bbox.transformed(fig.transFigure.inverted())
        x0 = float(bbox_fig.x0)
        y0 = float(bbox_fig.y0)
        w = float(bbox_fig.width)
        h = float(bbox_fig.height)
        return {
            "bbox_norm": {
                "x0": round(x0, 6),
                "y0": round(y0, 6),
                "width": round(w, 6),
                "height": round(h, 6),
            },
            "bbox_mm": {
                "x0": round(x0 * width_mm, 3),
                "y0": round(y0 * height_mm, 3),
                "width": round(w * width_mm, 3),
                "height": round(h * height_mm, 3),
            },
        }

    def _artist_bbox_payload(artist: Any) -> dict[str, Any] | None:
        if artist is None:
            return None
        visible = getattr(artist, "get_visible", lambda: True)()
        if not visible:
            return None
        text = getattr(artist, "get_text", lambda: "")()
        if isinstance(text, str) and not text.strip():
            return None
        bbox = artist.get_window_extent(renderer)
        return _display_payload(bbox)

    def _union_payload(artists: list[Any]) -> dict[str, Any] | None:
        boxes: list[Bbox] = []
        for artist in artists:
            if artist is None:
                continue
            visible = getattr(artist, "get_visible", lambda: True)()
            if not visible:
                continue
            text = getattr(artist, "get_text", lambda: "")()
            if isinstance(text, str) and not text.strip():
                continue
            boxes.append(artist.get_window_extent(renderer))
        if not boxes:
            return None
        return _display_payload(Bbox.union(boxes))

    axes_payload: list[dict[str, Any]] = []
    for idx, ax in enumerate(fig.axes):
        bbox = ax.get_position()
        title = ax.get_title()
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        raw_payload = _figure_norm_payload(Bbox.from_bounds(bbox.x0, bbox.y0, bbox.width, bbox.height))
        decorated_payload = _display_payload(ax.get_tightbbox(renderer))
        title_payload = _artist_bbox_payload(ax.title)
        xlabel_payload = _artist_bbox_payload(ax.xaxis.label)
        ylabel_payload = _artist_bbox_payload(ax.yaxis.label)
        xticks_payload = _union_payload(list(ax.get_xticklabels()))
        yticks_payload = _union_payload(list(ax.get_yticklabels()))
        legend = ax.get_legend()
        legend_payload = _display_payload(legend.get_window_extent(renderer)) if legend is not None else None
        axes_payload.append(
            {
                "index": idx,
                "kind": getattr(ax, "name", "axes"),
                "gid": ax.get_gid(),
                "has_data": bool(ax.has_data()),
                "title": title or None,
                "xlabel": xlabel or None,
                "ylabel": ylabel or None,
                **(raw_payload or {}),
                "decorated_bbox_norm": None if decorated_payload is None else decorated_payload["bbox_norm"],
                "decorated_bbox_mm": None if decorated_payload is None else decorated_payload["bbox_mm"],
                "title_bbox_norm": None if title_payload is None else title_payload["bbox_norm"],
                "title_bbox_mm": None if title_payload is None else title_payload["bbox_mm"],
                "xlabel_bbox_norm": None if xlabel_payload is None else xlabel_payload["bbox_norm"],
                "xlabel_bbox_mm": None if xlabel_payload is None else xlabel_payload["bbox_mm"],
                "ylabel_bbox_norm": None if ylabel_payload is None else ylabel_payload["bbox_norm"],
                "ylabel_bbox_mm": None if ylabel_payload is None else ylabel_payload["bbox_mm"],
                "xticklabels_bbox_norm": None if xticks_payload is None else xticks_payload["bbox_norm"],
                "xticklabels_bbox_mm": None if xticks_payload is None else xticks_payload["bbox_mm"],
                "yticklabels_bbox_norm": None if yticks_payload is None else yticks_payload["bbox_norm"],
                "yticklabels_bbox_mm": None if yticks_payload is None else yticks_payload["bbox_mm"],
                "legend_bbox_norm": None if legend_payload is None else legend_payload["bbox_norm"],
                "legend_bbox_mm": None if legend_payload is None else legend_payload["bbox_mm"],
            }
        )

    payload = {
        "contract_version": contract_version(),
        "figure_mm": {
            "width": round(float(width_mm), 3),
            "height": round(float(height_mm), 3),
        },
        "axes": axes_payload,
        "source_layout_spec": source_layout_spec(),
    }
    payload["typography_pt"] = typography or font_tokens()
    return payload


def _write_layout_metadata(
    fig: plt.Figure,
    out_prefix: str | Path,
    *,
    typography: dict[str, float] | None = None,
) -> Path:
    """Persist a sidecar JSON file with figure/axes layout metadata."""
    out_prefix = Path(out_prefix)
    layout_path = out_prefix.with_suffix(".layout.json")
    payload = collect_layout_metadata(fig, typography=typography)
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
