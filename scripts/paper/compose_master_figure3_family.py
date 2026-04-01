#!/usr/bin/env python3
"""Compose panel-first paper-facing figures (6-figure manuscript set).

This script rebuilds the manuscript assets for the current main-paper figure set
from panel-level assets whenever possible:

- Fig. 1: Paradigm Shift (5 panels: a,b fixed manual + c,d,e generated)
- Fig. 2: SVD Spectrum (7 panels: all generated as composite PDF)
- Fig. 3: Fingerprint Discriminability (5 panels: all generated)
- Fig. 4: Solver Mechanism (5 panels: a governed support schematic + b,c,d,e generated)
- Fig. 5: Performance + Structure (6 panels: all generated)
- Fig. 6: Universality (4 panels: a manual support + b,c,d generated)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

try:
    import fitz
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("PyMuPDF is required for figure composition") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.layout_contract import (
    contract_version,
    figure_section,
    font_pt,
    font_tokens,
    pt_to_px,
    source_layout_spec,
)


DEFAULT_PAPER_DIR = REPO_ROOT / "paper/figures"
MM_PER_INCH = 25.4
COMPOSE_DPI = 300
PANEL_LABEL_PT = font_pt("panel_label")
TYPOGRAPHY_PT = font_tokens()
ACTIVE_FIGURE_IDS = ("fig01", "fig02", "fig03", "fig04", "fig05", "fig06")

# --- Figure 1: Paradigm Shift (5 panels: a,b fixed manual + c,d,e generated) ---
FIG01_PANEL_A = REPO_ROOT / "figures/output/fig01_paradigm_shift_panels/fig01_panel_a_experimental_setup.png"
FIG01_PANEL_B = REPO_ROOT / "figures/output/fig01_paradigm_shift_panels/fig01_panel_b_spectral_fingerprint.png"
FIG01_COMPOSITE_CDE = REPO_ROOT / "figures/output/fig01_paradigm_data.pdf"
FIG01_COMPOSITE_CDE_LAYOUT = REPO_ROOT / "figures/output/fig01_paradigm_data.layout.json"
FIG01_WIDTH_MM = 183.0
FIG01_TOP_PANEL_WIDTH_MM = 89.0
FIG01_ROW_HEIGHT_MM = 65.0
FIG01_ROW_GAP_MM = 5.0
FIG01_HEIGHT_MM = FIG01_ROW_HEIGHT_MM * 2 + FIG01_ROW_GAP_MM
FIG01_PANEL_A_CROP = (0.00, 0.10, 1.00, 1.00)
FIG01_PANEL_B_CROP = (0.00, 0.10, 1.00, 1.00)

# --- Figure 2: SVD Spectrum (7 panels: all generated as composite PDF) ---
FIG02_COMPOSITE = REPO_ROOT / "figures/output/fig02_svd_spectrum.pdf"

# --- Figure 3: Fingerprint Discriminability (5 panels: all generated) ---
FIG03_COMPOSITE = REPO_ROOT / "figures/output/fig03_fingerprint_discriminability.pdf"

# --- Figure 4: Solver Mechanism (5 panels: a governed support schematic + b,c,d,e generated) ---
FIG04_PANEL_A = (
    REPO_ROOT
    / "figures/output/fig04_solver_dynamics_manuscript_panels/fig04_panel_a_architecture.jpg"
)
FIG04_PANEL_B = REPO_ROOT / "figures/output/fig04_solver_dynamics_panels/fig04_panel_b_gate_qk.pdf"
FIG04_PANEL_C = REPO_ROOT / "figures/output/fig04_solver_dynamics_panels/fig04_panel_c_update_residual.pdf"
FIG04_PANEL_D = REPO_ROOT / "figures/output/fig04_solver_dynamics_panels/fig04_panel_d_aggregation_bridge.pdf"
FIG04_PANEL_E = REPO_ROOT / "figures/output/fig04_solver_dynamics_panels/fig04_panel_e_ablation.pdf"
FIG04_COMPOSE = figure_section("fig04", "compose")
FIG04_WIDTH_MM = float(FIG04_COMPOSE["width_mm"])
FIG04_HEIGHT_MM = float(FIG04_COMPOSE["height_mm"])
FIG04_OUTER_MARGIN_MM = float(FIG04_COMPOSE["outer_margin_mm"])
FIG04_PANEL_A_SLOT_WIDTH_MM = float(FIG04_COMPOSE["panel_a_slot_width_mm"])
FIG04_PANEL_A_SLOT_HEIGHT_MM = float(FIG04_COMPOSE["panel_a_slot_height_mm"])
FIG04_ROW_GAP_MM = float(FIG04_COMPOSE["row_gap_mm"])
FIG04_COL_GAP_MM = float(FIG04_COMPOSE["col_gap_mm"])
FIG04_LOWER_PANEL_SLOT_WIDTH_MM = float(FIG04_COMPOSE["lower_panel_slot_width_mm"])
FIG04_LOWER_PANEL_SLOT_HEIGHT_MM = float(FIG04_COMPOSE["lower_panel_slot_height_mm"])
FIG04_PANEL_D_SLOT_HEIGHT_MM = float(FIG04_COMPOSE["panel_d_slot_height_mm"])
FIG04_PANEL_E_SLOT_HEIGHT_MM = float(FIG04_COMPOSE["panel_e_slot_height_mm"])
FIG04_RIGHT_STACK_GAP_MM = float(FIG04_COMPOSE["right_stack_gap_mm"])
FIG04_LABEL_LANE_MM = float(FIG04_COMPOSE["label_lane_mm"])
FIG04_CONTENT_INSET_X_MM = float(FIG04_COMPOSE["content_inset_x_mm"])
FIG04_CONTENT_INSET_BOTTOM_MM = float(FIG04_COMPOSE["content_inset_bottom_mm"])
FIG04_PANEL_A_CROP = (0.00, 0.00, 1.00, 1.00)

# --- Figure 5: Performance + Structure (6 panels: all generated) ---
FIG05_COMPOSITE = REPO_ROOT / "figures/output/fig05_performance_structure.pdf"
FIG05_COMPOSITE_LAYOUT = FIG05_COMPOSITE.with_suffix(".layout.json")
FIG05_WIDTH_MM = 183.0

# --- Figure 6: Universality (5 panels: a manual support + b,c,d,e generated) ---
FIG06_PANEL_A = REPO_ROOT / "figures/output/fig06_cross_material_universality_panels/fig06_panel_a_material_exemplars.png"
FIG06_PANEL_B = REPO_ROOT / "figures/output/fig06_universality_panels/fig06_panel_b_cross_material_h.pdf"
FIG06_PANEL_C = REPO_ROOT / "figures/output/fig06_universality_panels/fig06_panel_c_low_rank_continuity.pdf"
FIG06_PANEL_D = REPO_ROOT / "figures/output/fig06_universality_panels/fig06_panel_d_screening_consequence.pdf"
FIG06_PANEL_E = REPO_ROOT / "figures/output/fig06_universality_panels/fig06_panel_e_material_frequency_structure.pdf"
FIG06_LAYOUT_ERROR: Exception | None = None
try:
    FIG06_COMPOSE = figure_section("fig06", "compose")
    FIG06_WIDTH_MM = float(FIG06_COMPOSE["width_mm"])
    FIG06_HEIGHT_MM = float(FIG06_COMPOSE["height_mm"])
    FIG06_OUTER_MARGIN_X_MM = float(FIG06_COMPOSE["outer_margin_x_mm"])
    FIG06_OUTER_MARGIN_TOP_MM = float(FIG06_COMPOSE["outer_margin_top_mm"])
    FIG06_OUTER_MARGIN_BOTTOM_MM = float(FIG06_COMPOSE["outer_margin_bottom_mm"])
    FIG06_A_HEIGHT_MM = float(FIG06_COMPOSE["panel_a_height_mm"])
    FIG06_B_HEIGHT_MM = float(FIG06_COMPOSE["panel_b_height_mm"])
    FIG06_C_HEIGHT_MM = float(FIG06_COMPOSE["panel_c_height_mm"])
    FIG06_D_HEIGHT_MM = float(FIG06_COMPOSE["panel_d_height_mm"])
    FIG06_PANEL_C_WIDTH_MM = float(FIG06_COMPOSE["panel_c_width_mm"])
    FIG06_PANEL_E_WIDTH_MM = float(FIG06_COMPOSE["panel_e_width_mm"])
    FIG06_COL_GAP_MM = float(FIG06_COMPOSE["col_gap_mm"])
    FIG06_ROW_GAP_MM = float(FIG06_COMPOSE["row_gap_mm"])
except Exception as exc:  # pragma: no cover - only triggered by unrelated fig06 contract drift
    FIG06_LAYOUT_ERROR = exc
    FIG06_COMPOSE = {}
    FIG06_WIDTH_MM = FIG06_HEIGHT_MM = 0.0
    FIG06_OUTER_MARGIN_X_MM = FIG06_OUTER_MARGIN_TOP_MM = FIG06_OUTER_MARGIN_BOTTOM_MM = 0.0
    FIG06_A_HEIGHT_MM = FIG06_B_HEIGHT_MM = FIG06_C_HEIGHT_MM = FIG06_D_HEIGHT_MM = 0.0
    FIG06_PANEL_C_WIDTH_MM = FIG06_PANEL_E_WIDTH_MM = FIG06_COL_GAP_MM = 0.0
    FIG06_ROW_GAP_MM = 0.0
FIG06_PANEL_A_CROP = (0.015, 0.405, 0.985, 0.755)
FIG06_TYPOGRAPHY_PT = {
    **font_tokens(),
    **figure_section("fig06", "typography"),
}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for candidate in font_candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_unicode_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for candidate in font_candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _trim_white_border(img: Image.Image, padding: int = 8) -> Image.Image:
    trimmed, _bbox = _trim_white_border_with_bbox(img, padding=padding)
    return trimmed


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str = "#4A4A4A",
    width: int = 7,
    head_size: int = 18,
) -> None:
    draw.line([start, end], fill=fill, width=width)
    if start == end:
        return
    if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
        direction = 1 if end[0] >= start[0] else -1
        points = [
            end,
            (end[0] - direction * head_size, end[1] - head_size // 2),
            (end[0] - direction * head_size, end[1] + head_size // 2),
        ]
    else:
        direction = 1 if end[1] >= start[1] else -1
        points = [
            end,
            (end[0] - head_size // 2, end[1] - direction * head_size),
            (end[0] + head_size // 2, end[1] - direction * head_size),
        ]
    draw.polygon(points, fill=fill)


def _draw_centered_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    body: str | None = None,
    fill: str = "#F7F7F3",
    outline: str = "#C8C8C0",
    title_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    body_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    text_fill: str = "#1A1A1A",
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=26, fill=fill, outline=outline, width=4)
    lines = [title] if body is None else [title, body]
    fonts = [title_font] if body is None else [title_font, body_font]
    spacings = [10] if body is None else [10, 8]
    total_height = 0
    block_metrics: list[tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, tuple[int, int, int, int]]] = []
    for idx, (line, font) in enumerate(zip(lines, fonts)):
        bbox = draw.multiline_textbbox((0, 0), line, font=font, align="center", spacing=spacings[idx])
        block_metrics.append((line, font, bbox))
        total_height += bbox[3] - bbox[1]
        if idx < len(lines) - 1:
            total_height += 18
    cursor_y = y0 + (y1 - y0 - total_height) / 2
    for idx, (line, font, bbox) in enumerate(block_metrics):
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        cursor_x = x0 + (x1 - x0 - width) / 2
        draw.multiline_text(
            (cursor_x, cursor_y),
            line,
            font=font,
            fill=text_fill,
            align="center",
            spacing=spacings[idx],
        )
        cursor_y += height + 18


def _build_fig04_architecture_panel(panel_path: Path) -> None:
    """Generate the manuscript-facing Fig. 4a schematic from current solver logic."""
    width_px = 2784
    height_px = 1536
    active_height_px = int(round(height_px * 0.88))
    canvas = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = _load_font(66)
    stage_font = _load_font(46)
    box_title_font = _load_font(52)
    box_body_font = _load_unicode_font(38)

    box_colors = {
        "physics": ("#EEF5FB", "#2C7BC9"),
        "learned": ("#FBF1E8", "#D28A45"),
        "gate": ("#EDF8F3", "#12A36E"),
        "residual": ("#F3F4F6", "#7A7F85"),
        "update": ("#F1F5FB", "#3B82F6"),
        "readout": ("#F7F3FB", "#8B5FBF"),
    }
    arrow_color = "#4B4B4B"

    draw.text((width_px * 0.34, 70), "Stage t", fill="#1A1A1A", font=title_font)
    draw.text((width_px * 0.76, 70), "Readout / residual branch", fill="#1A1A1A", font=stage_font)
    draw.line((170, 150, width_px - 160, 150), fill="#202020", width=5)
    _draw_arrow(draw, (width_px - 220, 150), (width_px - 140, 150), fill="#202020", width=5, head_size=16)

    boxes = {
        "residual": (110, 360, 500, 620),
        "correlation": (580, 360, 1020, 620),
        "scores": (1100, 360, 1540, 620),
        "gate": (1620, 360, 1890, 620),
        "update": (1970, 360, 2400, 620),
        "readout": (1860, 150, 2250, 330),
        "doa": (2320, 150, 2700, 330),
    }

    _draw_centered_box(
        draw,
        boxes["residual"],
        title="Residual",
        body="r_t",
        fill=box_colors["residual"][0],
        outline=box_colors["residual"][1],
        title_font=box_title_font,
        body_font=box_body_font,
    )
    _draw_centered_box(
        draw,
        boxes["correlation"],
        title="Physical correlation",
        body="g_t = D^T r_t",
        fill=box_colors["physics"][0],
        outline=box_colors["physics"][1],
        title_font=box_title_font,
        body_font=box_body_font,
    )
    _draw_centered_box(
        draw,
        boxes["scores"],
        title="Expert scores",
        body="s_t[e] = <q_t, k_e>/sqrt(d_k)",
        fill=box_colors["learned"][0],
        outline=box_colors["learned"][1],
        title_font=box_title_font,
        body_font=box_body_font,
    )
    _draw_centered_box(
        draw,
        boxes["gate"],
        title="Routed gate",
        body="w_t",
        fill=box_colors["gate"][0],
        outline=box_colors["gate"][1],
        title_font=box_title_font,
        body_font=box_body_font,
    )
    _draw_centered_box(
        draw,
        boxes["update"],
        title="Sparse / residual update",
        body="Delta x_t = w_t * g_t\nr_(t+1) = r_t - D(eta Delta x_t)",
        fill=box_colors["update"][0],
        outline=box_colors["update"][1],
        title_font=_load_font(46),
        body_font=_load_unicode_font(30),
    )
    _draw_centered_box(
        draw,
        boxes["readout"],
        title="Expert-score readout",
        body="s_bar[e]",
        fill=box_colors["readout"][0],
        outline=box_colors["readout"][1],
        title_font=box_title_font,
        body_font=box_body_font,
    )
    _draw_centered_box(
        draw,
        boxes["doa"],
        title="Final DOA",
        body="theta_hat = theta_argmax_e\ns_bar[e]",
        fill="#F8F8F8",
        outline="#808080",
        title_font=box_title_font,
        body_font=box_body_font,
    )

    # Main stage-t path
    _draw_arrow(draw, (500, 490), (580, 490), fill=arrow_color)
    _draw_arrow(draw, (1020, 490), (1100, 490), fill=arrow_color)
    _draw_arrow(draw, (1540, 490), (1620, 490), fill=arrow_color)
    _draw_arrow(draw, (1890, 490), (1970, 490), fill=arrow_color)

    # Readout branch
    _draw_arrow(draw, (1320, 360), (1320, 240), fill=arrow_color)
    _draw_arrow(draw, (2250, 240), (2320, 240), fill=arrow_color)

    _ensure_parent(panel_path)
    canvas.save(panel_path, quality=95)


def _trim_white_border_with_bbox(img: Image.Image, padding: int = 8) -> tuple[Image.Image, tuple[int, int, int, int]]:
    rgb = ImageOps.exif_transpose(img).convert("RGB")
    bg = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return rgb, (0, 0, rgb.width, rgb.height)
    left = max(bbox[0] - padding, 0)
    top = max(bbox[1] - padding, 0)
    right = min(bbox[2] + padding, rgb.width)
    bottom = min(bbox[3] + padding, rgb.height)
    return rgb.crop((left, top, right, bottom)), (left, top, right, bottom)


def _render_pdf(pdf_path: Path, scale: float = 3.0) -> Image.Image:
    with fitz.open(str(pdf_path)) as doc:
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _resize_to_width(img: Image.Image, target_width: int) -> Image.Image:
    scale = target_width / max(img.width, 1)
    target_height = int(round(img.height * scale))
    return img.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _contain_in_box(img: Image.Image, box_width: int, box_height: int) -> Image.Image:
    contained, _geometry = _contain_in_box_with_geometry(img, box_width, box_height)
    return contained


def _contain_in_box_with_geometry(
    img: Image.Image,
    box_width: int,
    box_height: int,
) -> tuple[Image.Image, dict[str, int | float]]:
    scale = min(box_width / max(img.width, 1), box_height / max(img.height, 1))
    target_width = int(round(img.width * scale))
    target_height = int(round(img.height * scale))
    resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (box_width, box_height), "white")
    x = (box_width - target_width) // 2
    y = (box_height - target_height) // 2
    canvas.paste(resized, (x, y))
    return canvas, {
        "offset_x_px": x,
        "offset_y_px": y,
        "resized_width_px": target_width,
        "resized_height_px": target_height,
        "scale": scale,
    }


def _crop_relative(
    img: Image.Image,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> Image.Image:
    width, height = img.size
    box = (
        int(round(width * left)),
        int(round(height * top)),
        int(round(width * right)),
        int(round(height * bottom)),
    )
    return img.crop(box)


def _reorder_strip_tiles(img: Image.Image, order: list[int]) -> Image.Image:
    if not order:
        return img
    n_tiles = len(order)
    tile_width = img.width // n_tiles
    tiles = []
    for idx in range(n_tiles):
        left = idx * tile_width
        right = img.width if idx == n_tiles - 1 else (idx + 1) * tile_width
        tiles.append(img.crop((left, 0, right, img.height)))
    canvas = Image.new("RGB", img.size, "white")
    x = 0
    for idx in order:
        tile = tiles[idx]
        canvas.paste(tile, (x, 0))
        x += tile.width
    return canvas


def _mm_to_px(mm: float) -> int:
    return int(round(mm / MM_PER_INCH * COMPOSE_DPI))


def _normalize_figure_ids(raw: str | None) -> list[str]:
    if not raw:
        return list(ACTIVE_FIGURE_IDS)
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in requested if item not in ACTIVE_FIGURE_IDS]
    if invalid:
        raise ValueError(f"Unsupported figure id(s): {', '.join(invalid)}")
    ordered: list[str] = []
    seen: set[str] = set()
    for figure_id in requested:
        if figure_id in seen:
            continue
        seen.add(figure_id)
        ordered.append(figure_id)
    return ordered


def _bbox_payload(x0_mm: float, y0_mm: float, width_mm: float, height_mm: float, figure_width_mm: float, figure_height_mm: float) -> dict[str, dict[str, float]]:
    return {
        "bbox_mm": {
            "x0": round(float(x0_mm), 3),
            "y0": round(float(y0_mm), 3),
            "width": round(float(width_mm), 3),
            "height": round(float(height_mm), 3),
        },
        "bbox_norm": {
            "x0": round(float(x0_mm / figure_width_mm), 6),
            "y0": round(float(y0_mm / figure_height_mm), 6),
            "width": round(float(width_mm / figure_width_mm), 6),
            "height": round(float(height_mm / figure_height_mm), 6),
        },
    }


def _write_layout_metadata(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_reference_panel_manifest(
    path: Path,
    *,
    figure_id: str,
    composite_asset: str,
    panels: list[dict[str, str]],
) -> None:
    _ensure_parent(path)
    payload = {
        "composite_asset": composite_asset,
        "figure_id": figure_id,
        "panel_order": [panel["panel_id"] for panel in panels],
        "panels": [
            {
                "asset_path": panel["asset_path"],
                "panel_id": panel["panel_id"],
                "provenance_mode": panel["provenance_mode"],
                "source_asset": panel["asset_path"],
                "source_layer": "mixed_panel_lineage",
                "storage_mode": "reference_existing_outputs",
                "title": panel["title"],
            }
            for panel in panels
        ],
        "storage_mode": "reference_existing_outputs",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _transform_bbox_norm_for_manuscript(
    bbox_norm: dict[str, float] | None,
    *,
    source_width_px: int,
    source_height_px: int,
    crop_box: tuple[int, int, int, int],
    scale: float,
    offset_x_px: int,
    offset_y_px: int,
    canvas_width_px: int,
    canvas_height_px: int,
    figure_width_mm: float,
    figure_height_mm: float,
) -> tuple[dict[str, float], dict[str, float]] | tuple[None, None]:
    if bbox_norm is None:
        return None, None

    crop_left, crop_top, _crop_right, _crop_bottom = crop_box
    left_px = float(bbox_norm["x0"]) * source_width_px
    width_px = float(bbox_norm["width"]) * source_width_px
    right_px = left_px + width_px

    bottom_from_bottom_px = float(bbox_norm["y0"]) * source_height_px
    height_px = float(bbox_norm["height"]) * source_height_px
    top_px = source_height_px - (bottom_from_bottom_px + height_px)
    bottom_px = top_px + height_px

    left_canvas_px = (left_px - crop_left) * scale + offset_x_px
    right_canvas_px = (right_px - crop_left) * scale + offset_x_px
    top_canvas_px = (top_px - crop_top) * scale + offset_y_px
    bottom_canvas_px = (bottom_px - crop_top) * scale + offset_y_px

    width_canvas_px = right_canvas_px - left_canvas_px
    height_canvas_px = bottom_canvas_px - top_canvas_px
    y0_bottom_canvas_px = canvas_height_px - bottom_canvas_px

    norm = {
        "x0": round(left_canvas_px / canvas_width_px, 6),
        "y0": round(y0_bottom_canvas_px / canvas_height_px, 6),
        "width": round(width_canvas_px / canvas_width_px, 6),
        "height": round(height_canvas_px / canvas_height_px, 6),
    }
    mm = {
        "x0": round(norm["x0"] * figure_width_mm, 3),
        "y0": round(norm["y0"] * figure_height_mm, 3),
        "width": round(norm["width"] * figure_width_mm, 3),
        "height": round(norm["height"] * figure_height_mm, 3),
    }
    return norm, mm


def _transform_layout_payload_for_manuscript(
    payload: dict[str, Any],
    *,
    source_width_px: int,
    source_height_px: int,
    crop_box: tuple[int, int, int, int],
    resized_width_px: int,
    offset_x_px: int,
    offset_y_px: int,
    canvas_width_px: int,
    canvas_height_px: int,
    figure_width_mm: float,
    figure_height_mm: float | None = None,
) -> dict[str, Any]:
    if figure_height_mm is None:
        figure_height_mm = round(figure_width_mm * canvas_height_px / canvas_width_px, 3)
    cropped_width_px = max(crop_box[2] - crop_box[0], 1)
    scale = resized_width_px / cropped_width_px
    bbox_keys = (
        "bbox",
        "decorated_bbox",
        "title_bbox",
        "xlabel_bbox",
        "ylabel_bbox",
        "xticklabels_bbox",
        "yticklabels_bbox",
        "legend_bbox",
    )

    axes_payload: list[dict[str, Any]] = []
    for axis in payload.get("axes", []):
        entry = dict(axis)
        for key in bbox_keys:
            norm_key = f"{key}_norm"
            mm_key = f"{key}_mm"
            norm, mm = _transform_bbox_norm_for_manuscript(
                axis.get(norm_key),
                source_width_px=source_width_px,
                source_height_px=source_height_px,
                crop_box=crop_box,
                scale=scale,
                offset_x_px=offset_x_px,
                offset_y_px=offset_y_px,
                canvas_width_px=canvas_width_px,
                canvas_height_px=canvas_height_px,
                figure_width_mm=figure_width_mm,
                figure_height_mm=figure_height_mm,
            )
            entry[norm_key] = norm
            entry[mm_key] = mm
        axes_payload.append(entry)

    return {
        "contract_version": payload.get("contract_version", contract_version()),
        "figure_mm": {"width": figure_width_mm, "height": figure_height_mm},
        "axes": axes_payload,
        "source_layout_spec": payload.get("source_layout_spec", source_layout_spec()),
        "typography_pt": payload.get("typography_pt", TYPOGRAPHY_PT),
    }


def _bbox_payload_from_canvas_px(
    left_px: int,
    top_px: int,
    width_px: int,
    height_px: int,
    *,
    canvas_width_px: int,
    canvas_height_px: int,
    figure_width_mm: float,
    figure_height_mm: float,
) -> dict[str, dict[str, float]]:
    x0_norm = float(left_px) / canvas_width_px
    y0_norm = float(canvas_height_px - (top_px + height_px)) / canvas_height_px
    width_norm = float(width_px) / canvas_width_px
    height_norm = float(height_px) / canvas_height_px
    return {
        "bbox_norm": {
            "x0": round(x0_norm, 6),
            "y0": round(y0_norm, 6),
            "width": round(width_norm, 6),
            "height": round(height_norm, 6),
        },
        "bbox_mm": {
            "x0": round(x0_norm * figure_width_mm, 3),
            "y0": round(y0_norm * figure_height_mm, 3),
            "width": round(width_norm * figure_width_mm, 3),
            "height": round(height_norm * figure_height_mm, 3),
        },
    }


def _draw_panel_label(draw: ImageDraw.ImageDraw, label: str, x_px: int, y_px: int) -> None:
    panel_label_px = pt_to_px(PANEL_LABEL_PT, COMPOSE_DPI)
    draw.text((x_px, y_px), label, fill="black", font=_load_font(panel_label_px))


def _draw_boxed_panel_label(draw: ImageDraw.ImageDraw, label: str, x_px: int, y_px: int) -> None:
    panel_label_px = pt_to_px(PANEL_LABEL_PT, COMPOSE_DPI)
    font = _load_font(panel_label_px)
    left, top, right, bottom = draw.textbbox((x_px, y_px), label, font=font)
    pad_x = max(4, panel_label_px // 5)
    pad_y = max(3, panel_label_px // 8)
    draw.rounded_rectangle(
        (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y),
        radius=6,
        fill="white",
        outline="#404040",
        width=1,
    )
    draw.text((x_px, y_px), label, fill="black", font=font)


def _save_composite(img: Image.Image, path: Path) -> None:
    _ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        img.save(path, format="JPEG", quality=95, subsampling=0, dpi=(COMPOSE_DPI, COMPOSE_DPI))
    elif suffix == ".png":
        img.save(path, format="PNG", dpi=(COMPOSE_DPI, COMPOSE_DPI))
    else:
        raise ValueError(f"Unsupported composite suffix for {path}")


def compose_fig01(paper_dir: Path) -> list[Path]:
    """Fig 1: mm-driven layout with fixed a,b panels and the generated c,d,e strip."""
    fig01_asset = paper_dir / "fig01_paradigm-shift.jpg"
    fig01_layout_asset = fig01_asset.with_suffix(".layout.json")

    panel_a = ImageOps.exif_transpose(Image.open(FIG01_PANEL_A)).convert("RGB")
    panel_b = ImageOps.exif_transpose(Image.open(FIG01_PANEL_B)).convert("RGB")
    panel_a = _crop_relative(panel_a, *FIG01_PANEL_A_CROP)
    panel_b = _crop_relative(panel_b, *FIG01_PANEL_B_CROP)
    panel_cde = _render_pdf(FIG01_COMPOSITE_CDE, scale=4.0).convert("RGB")

    figure_width_px = _mm_to_px(FIG01_WIDTH_MM)
    top_panel_width_px = _mm_to_px(FIG01_TOP_PANEL_WIDTH_MM)
    top_gap_px = figure_width_px - 2 * top_panel_width_px
    row_height_px = _mm_to_px(FIG01_ROW_HEIGHT_MM)
    row_gap_px = _mm_to_px(FIG01_ROW_GAP_MM)
    figure_height_px = row_height_px * 2 + row_gap_px

    panel_a = _contain_in_box(panel_a, top_panel_width_px, row_height_px)
    panel_b = _contain_in_box(panel_b, top_panel_width_px, row_height_px)
    panel_cde = _resize_to_width(panel_cde, figure_width_px)

    canvas = Image.new("RGB", (figure_width_px, figure_height_px), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(panel_a, (0, 0))
    canvas.paste(panel_b, (top_panel_width_px + top_gap_px, 0))
    _draw_panel_label(draw, "a", _mm_to_px(2.5), _mm_to_px(2.0))
    _draw_panel_label(draw, "b", top_panel_width_px + top_gap_px + _mm_to_px(2.5), _mm_to_px(2.0))

    bottom_row_y_px = row_height_px + row_gap_px
    bottom_strip_y_px = bottom_row_y_px + max((row_height_px - panel_cde.height) // 2, 0)
    canvas.paste(panel_cde, (0, bottom_strip_y_px))

    _save_composite(canvas, fig01_asset)

    composite_layout = json.loads(FIG01_COMPOSITE_CDE_LAYOUT.read_text(encoding="utf-8"))
    bottom_scale = FIG01_WIDTH_MM / composite_layout["figure_mm"]["width"]
    bottom_strip_height_mm = composite_layout["figure_mm"]["height"] * bottom_scale
    bottom_row_offset_mm = max((FIG01_ROW_HEIGHT_MM - bottom_strip_height_mm) / 2.0, 0.0)

    axes = [
        {
            "index": 0,
            "panel_id": "a",
            "kind": "manual",
            "has_data": False,
            "title": "Experimental setup",
            **_bbox_payload(
                x0_mm=0.0,
                y0_mm=FIG01_ROW_HEIGHT_MM + FIG01_ROW_GAP_MM,
                width_mm=FIG01_TOP_PANEL_WIDTH_MM,
                height_mm=FIG01_ROW_HEIGHT_MM,
                figure_width_mm=FIG01_WIDTH_MM,
                figure_height_mm=FIG01_HEIGHT_MM,
            ),
        },
        {
            "index": 1,
            "panel_id": "b",
            "kind": "manual",
            "has_data": False,
            "title": "Physical mechanism schematic",
            **_bbox_payload(
                x0_mm=FIG01_WIDTH_MM - FIG01_TOP_PANEL_WIDTH_MM,
                y0_mm=FIG01_ROW_HEIGHT_MM + FIG01_ROW_GAP_MM,
                width_mm=FIG01_TOP_PANEL_WIDTH_MM,
                height_mm=FIG01_ROW_HEIGHT_MM,
                figure_width_mm=FIG01_WIDTH_MM,
                figure_height_mm=FIG01_HEIGHT_MM,
            ),
        },
    ]

    panel_ids = ("c", "d", "e")
    for ax_meta, panel_id in zip(composite_layout.get("axes", []), panel_ids, strict=True):
        bbox = ax_meta["bbox_mm"]
        x0_mm = bbox["x0"] * bottom_scale
        y0_mm = bbox["y0"] * bottom_scale + bottom_row_offset_mm
        width_mm = bbox["width"] * bottom_scale
        height_mm = bbox["height"] * bottom_scale
        axes.append(
            {
                "index": len(axes),
                "panel_id": panel_id,
                "kind": ax_meta.get("kind"),
                "has_data": ax_meta.get("has_data", True),
                "title": ax_meta.get("title"),
                "xlabel": ax_meta.get("xlabel"),
                "ylabel": ax_meta.get("ylabel"),
                **_bbox_payload(
                    x0_mm=x0_mm,
                    y0_mm=y0_mm,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    figure_width_mm=FIG01_WIDTH_MM,
                    figure_height_mm=FIG01_HEIGHT_MM,
                ),
            }
        )

    _write_layout_metadata(
        fig01_layout_asset,
        {
            "contract_version": contract_version(),
            "figure_mm": {
                "width": FIG01_WIDTH_MM,
                "height": FIG01_HEIGHT_MM,
            },
            "axes": axes,
            "source_layout_spec": source_layout_spec(),
            "typography_pt": FIG06_TYPOGRAPHY_PT,
        },
    )
    return [fig01_asset, fig01_layout_asset]


def compose_fig02(paper_dir: Path) -> list[Path]:
    """Fig 2: All 6 panels from composite PDF (a-f)."""
    fig02_asset = paper_dir / "fig02_svd-physical-dictionary.jpg"

    composite = _trim_white_border(_render_pdf(FIG02_COMPOSITE, scale=4.0), padding=4)
    target_width = 2200
    composite = _resize_to_width(composite, target_width)

    margin = 40
    canvas_w = margin * 2 + composite.width
    canvas_h = margin * 2 + composite.height
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(composite, (margin, margin))

    _save_composite(canvas, fig02_asset)
    return [fig02_asset]


def compose_fig03(paper_dir: Path) -> list[Path]:
    """Fig 3: All 5 panels from composite PDF (a-e)."""
    fig03_asset = paper_dir / "fig03_fingerprint-discriminability.jpg"

    composite = _trim_white_border(_render_pdf(FIG03_COMPOSITE, scale=4.0), padding=4)
    target_width = 2200
    composite = _resize_to_width(composite, target_width)

    margin = 40
    canvas_w = margin * 2 + composite.width
    canvas_h = margin * 2 + composite.height
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(composite, (margin, margin))

    _save_composite(canvas, fig03_asset)
    return [fig03_asset]


def compose_fig04(paper_dir: Path) -> list[Path]:
    """Fig 4: hero architecture plus b/c lower panels and a stacked d/e right slot."""
    fig04_asset = paper_dir / "fig04_solver-dynamics.jpg"
    fig04_layout_asset = fig04_asset.with_suffix(".layout.json")

    _build_fig04_architecture_panel(FIG04_PANEL_A)
    panel_a = _trim_white_border(Image.open(FIG04_PANEL_A).convert("RGB"), padding=0)
    panel_a = _crop_relative(panel_a, *FIG04_PANEL_A_CROP)
    figure_width_px = _mm_to_px(FIG04_WIDTH_MM)
    figure_height_px = _mm_to_px(FIG04_HEIGHT_MM)
    outer_margin_px = _mm_to_px(FIG04_OUTER_MARGIN_MM)
    a_slot_width_px = _mm_to_px(FIG04_PANEL_A_SLOT_WIDTH_MM)
    a_slot_height_px = _mm_to_px(FIG04_PANEL_A_SLOT_HEIGHT_MM)
    row_gap_px = _mm_to_px(FIG04_ROW_GAP_MM)
    col_gap_px = _mm_to_px(FIG04_COL_GAP_MM)
    lower_slot_width_px = _mm_to_px(FIG04_LOWER_PANEL_SLOT_WIDTH_MM)
    lower_slot_height_px = _mm_to_px(FIG04_LOWER_PANEL_SLOT_HEIGHT_MM)
    d_slot_height_px = _mm_to_px(FIG04_PANEL_D_SLOT_HEIGHT_MM)
    e_slot_height_px = _mm_to_px(FIG04_PANEL_E_SLOT_HEIGHT_MM)
    right_stack_gap_px = _mm_to_px(FIG04_RIGHT_STACK_GAP_MM)
    label_lane_px = _mm_to_px(FIG04_LABEL_LANE_MM)
    content_inset_x_px = _mm_to_px(FIG04_CONTENT_INSET_X_MM)
    content_inset_bottom_px = _mm_to_px(FIG04_CONTENT_INSET_BOTTOM_MM)

    a_slot_left_px = outer_margin_px
    a_slot_top_px = outer_margin_px
    lower_slot_top_px = outer_margin_px + a_slot_height_px + row_gap_px
    b_slot_left_px = outer_margin_px
    c_slot_left_px = b_slot_left_px + lower_slot_width_px + col_gap_px
    d_slot_left_px = figure_width_px - outer_margin_px - lower_slot_width_px
    d_slot_top_px = lower_slot_top_px
    e_slot_left_px = d_slot_left_px
    e_slot_top_px = d_slot_top_px + d_slot_height_px + right_stack_gap_px

    def _content_box(slot_left_px: int, slot_top_px: int, slot_width_px: int, slot_height_px: int) -> tuple[int, int, int, int]:
        return (
            slot_left_px + content_inset_x_px,
            slot_top_px + label_lane_px,
            slot_width_px - 2 * content_inset_x_px,
            slot_height_px - label_lane_px - content_inset_bottom_px,
        )

    def _paste_panel_pdf(
        pdf_path: Path,
        *,
        padding: int,
        content_left_px: int,
        content_top_px: int,
        content_width_px: int,
        content_height_px: int,
    ) -> tuple[Image.Image, list[dict[str, Any]]]:
        rendered = _render_pdf(pdf_path, scale=4.0)
        trimmed, crop_box = _trim_white_border_with_bbox(rendered, padding=padding)
        contained, geometry = _contain_in_box_with_geometry(trimmed, content_width_px, content_height_px)
        layout_payload = json.loads(pdf_path.with_suffix(".layout.json").read_text(encoding="utf-8"))
        transformed = _transform_layout_payload_for_manuscript(
            layout_payload,
            source_width_px=rendered.width,
            source_height_px=rendered.height,
            crop_box=crop_box,
            resized_width_px=int(geometry["resized_width_px"]),
            offset_x_px=content_left_px + int(geometry["offset_x_px"]),
            offset_y_px=content_top_px + int(geometry["offset_y_px"]),
            canvas_width_px=figure_width_px,
            canvas_height_px=figure_height_px,
            figure_width_mm=FIG04_WIDTH_MM,
            figure_height_mm=FIG04_HEIGHT_MM,
        )
        return contained, list(transformed["axes"])

    canvas = Image.new("RGB", (figure_width_px, figure_height_px), "white")
    draw = ImageDraw.Draw(canvas)

    a_content_left_px, a_content_top_px, a_content_width_px, a_content_height_px = _content_box(
        a_slot_left_px,
        a_slot_top_px,
        a_slot_width_px,
        a_slot_height_px,
    )
    panel_a, panel_a_geometry = _contain_in_box_with_geometry(
        panel_a,
        a_content_width_px,
        a_content_height_px,
    )
    canvas.paste(panel_a, (a_content_left_px, a_content_top_px))

    b_content_left_px, b_content_top_px, b_content_width_px, b_content_height_px = _content_box(
        b_slot_left_px,
        lower_slot_top_px,
        lower_slot_width_px,
        lower_slot_height_px,
    )
    c_content_left_px, c_content_top_px, c_content_width_px, c_content_height_px = _content_box(
        c_slot_left_px,
        lower_slot_top_px,
        lower_slot_width_px,
        lower_slot_height_px,
    )
    d_content_left_px, d_content_top_px, d_content_width_px, d_content_height_px = _content_box(
        d_slot_left_px,
        d_slot_top_px,
        lower_slot_width_px,
        d_slot_height_px,
    )
    e_content_left_px, e_content_top_px, e_content_width_px, e_content_height_px = _content_box(
        e_slot_left_px,
        e_slot_top_px,
        lower_slot_width_px,
        e_slot_height_px,
    )

    panel_b, panel_b_axes = _paste_panel_pdf(
        FIG04_PANEL_B,
        padding=10,
        content_left_px=b_content_left_px,
        content_top_px=b_content_top_px,
        content_width_px=b_content_width_px,
        content_height_px=b_content_height_px,
    )
    panel_c, panel_c_axes = _paste_panel_pdf(
        FIG04_PANEL_C,
        padding=10,
        content_left_px=c_content_left_px,
        content_top_px=c_content_top_px,
        content_width_px=c_content_width_px,
        content_height_px=c_content_height_px,
    )
    panel_d, panel_d_axes = _paste_panel_pdf(
        FIG04_PANEL_D,
        padding=8,
        content_left_px=d_content_left_px,
        content_top_px=d_content_top_px,
        content_width_px=d_content_width_px,
        content_height_px=d_content_height_px,
    )
    panel_e, panel_e_axes = _paste_panel_pdf(
        FIG04_PANEL_E,
        padding=8,
        content_left_px=e_content_left_px,
        content_top_px=e_content_top_px,
        content_width_px=e_content_width_px,
        content_height_px=e_content_height_px,
    )

    canvas.paste(panel_b, (b_content_left_px, b_content_top_px))
    canvas.paste(panel_c, (c_content_left_px, c_content_top_px))
    canvas.paste(panel_d, (d_content_left_px, d_content_top_px))
    canvas.paste(panel_e, (e_content_left_px, e_content_top_px))

    label_x_px = _mm_to_px(1.5)
    label_y_px = _mm_to_px(0.6)
    _draw_panel_label(draw, "a", a_slot_left_px + label_x_px, a_slot_top_px + label_y_px)
    _draw_panel_label(draw, "b", b_slot_left_px + label_x_px, lower_slot_top_px + label_y_px)
    _draw_panel_label(draw, "c", c_slot_left_px + label_x_px, lower_slot_top_px + label_y_px)
    _draw_panel_label(draw, "d", d_slot_left_px + label_x_px, d_slot_top_px + label_y_px)
    _draw_panel_label(draw, "e", e_slot_left_px + label_x_px, e_slot_top_px + label_y_px)

    _save_composite(canvas, fig04_asset)
    manuscript_manifest = REPO_ROOT / "figures/output/fig04_solver_dynamics_manuscript_panels/fig04_panel_manifest.json"
    _write_reference_panel_manifest(
        manuscript_manifest,
        figure_id="fig04",
        composite_asset="paper/figures/fig04_solver-dynamics.jpg",
        panels=[
            {
                "panel_id": "a",
                "title": "Architecture diagram",
                "asset_path": "figures/output/fig04_solver_dynamics_manuscript_panels/fig04_panel_a_architecture.jpg",
                "provenance_mode": "manual_support",
            },
            {
                "panel_id": "b",
                "title": "Routing formation",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_b_gate_qk.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "c",
                "title": "Gated update and residual correction",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_c_update_residual.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "d",
                "title": "Aggregation bridge",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_d_aggregation_bridge.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "e",
                "title": "Routing-mechanism ablation",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_e_ablation.pdf",
                "provenance_mode": "data_backed",
            },
        ],
    )
    a_bbox = _bbox_payload_from_canvas_px(
        a_content_left_px + int(panel_a_geometry["offset_x_px"]),
        a_content_top_px + int(panel_a_geometry["offset_y_px"]),
        int(panel_a_geometry["resized_width_px"]),
        int(panel_a_geometry["resized_height_px"]),
        canvas_width_px=figure_width_px,
        canvas_height_px=figure_height_px,
        figure_width_mm=FIG04_WIDTH_MM,
        figure_height_mm=FIG04_HEIGHT_MM,
    )
    a_slot_bbox = _bbox_payload_from_canvas_px(
        a_slot_left_px,
        a_slot_top_px,
        a_slot_width_px,
        a_slot_height_px,
        canvas_width_px=figure_width_px,
        canvas_height_px=figure_height_px,
        figure_width_mm=FIG04_WIDTH_MM,
        figure_height_mm=FIG04_HEIGHT_MM,
    )
    axes_payload: list[dict[str, Any]] = [
        {
            "index": 0,
            "panel_id": "a",
            "gid": "fig04.panel_a.main",
            "kind": "manual",
            "has_data": False,
            "title": "Architecture diagram",
            **a_bbox,
            "decorated_bbox_norm": a_slot_bbox["bbox_norm"],
            "decorated_bbox_mm": a_slot_bbox["bbox_mm"],
            "title_bbox_norm": None,
            "title_bbox_mm": None,
            "xlabel": None,
            "xlabel_bbox_norm": None,
            "xlabel_bbox_mm": None,
            "ylabel": None,
            "ylabel_bbox_norm": None,
            "ylabel_bbox_mm": None,
            "xticklabels_bbox_norm": None,
            "xticklabels_bbox_mm": None,
            "yticklabels_bbox_norm": None,
            "yticklabels_bbox_mm": None,
            "legend_bbox_norm": None,
            "legend_bbox_mm": None,
        }
    ]
    axes_payload.extend(panel_b_axes)
    axes_payload.extend(panel_c_axes)
    axes_payload.extend(panel_d_axes)
    axes_payload.extend(panel_e_axes)
    for idx, axis in enumerate(axes_payload):
        gid = axis.get("gid")
        if gid:
            parts = str(gid).split(".")
            if len(parts) >= 2 and parts[1].startswith("panel_"):
                axis.setdefault("panel_id", parts[1].removeprefix("panel_"))
        axis["index"] = idx
    _write_layout_metadata(
        fig04_layout_asset,
        {
            "contract_version": contract_version(),
            "figure_mm": {"width": FIG04_WIDTH_MM, "height": FIG04_HEIGHT_MM},
            "axes": axes_payload,
            "source_layout_spec": source_layout_spec(),
            "typography_pt": TYPOGRAPHY_PT,
        },
    )
    return [fig04_asset, fig04_layout_asset]


def compose_fig05(paper_dir: Path) -> list[Path]:
    """Fig 5: All 5 panels from composite PDF (a-e)."""
    fig05_asset = paper_dir / "fig05_performance-structure.jpg"
    fig05_layout_asset = fig05_asset.with_suffix(".layout.json")

    rendered = _render_pdf(FIG05_COMPOSITE, scale=4.0)
    composite, crop_box = _trim_white_border_with_bbox(rendered, padding=4)
    target_width = 2200
    composite = _resize_to_width(composite, target_width)

    # Keep the same left/right breathing room while trimming the vertical collar
    # slightly so the final manuscript asset stays under the 170 mm main-figure cap.
    margin_x = 40
    margin_y = 36
    canvas_w = margin_x * 2 + composite.width
    canvas_h = margin_y * 2 + composite.height
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(composite, (margin_x, margin_y))

    _save_composite(canvas, fig05_asset)
    if FIG05_COMPOSITE_LAYOUT.exists():
        composite_layout = json.loads(FIG05_COMPOSITE_LAYOUT.read_text(encoding="utf-8"))
        manuscript_layout = _transform_layout_payload_for_manuscript(
            composite_layout,
            source_width_px=rendered.width,
            source_height_px=rendered.height,
            crop_box=crop_box,
            resized_width_px=composite.width,
            offset_x_px=margin_x,
            offset_y_px=margin_y,
            canvas_width_px=canvas_w,
            canvas_height_px=canvas_h,
            figure_width_mm=FIG05_WIDTH_MM,
        )
        _write_layout_metadata(fig05_layout_asset, manuscript_layout)
        return [fig05_asset, fig05_layout_asset]
    return [fig05_asset]


def compose_fig06(paper_dir: Path) -> list[Path]:
    """Fig 6: manual exemplar strip + H breadth + low-rank + mechanism + consequence."""
    if FIG06_LAYOUT_ERROR is not None:
        raise RuntimeError(
            "Fig. 6 compose contract is invalid in the current worktree; "
            "fix the fig06 layout contract before composing fig06"
        ) from FIG06_LAYOUT_ERROR

    fig06_asset = paper_dir / "fig06_universality.jpg"
    fig06_layout_asset = fig06_asset.with_suffix(".layout.json")
    manuscript_panel_dir = REPO_ROOT / "figures/output/fig06_universality_manuscript_panels"
    panel_a_manuscript_asset = manuscript_panel_dir / "fig06_panel_a_material_exemplars.png"

    if not FIG06_PANEL_A.exists():
        raise FileNotFoundError(f"Missing Fig. 6 panel a support asset: {FIG06_PANEL_A}")
    panel_a = Image.open(FIG06_PANEL_A).convert("RGB")
    panel_a = _trim_white_border(panel_a, padding=0)
    panel_a = _crop_relative(panel_a, *FIG06_PANEL_A_CROP)
    panel_a = _reorder_strip_tiles(panel_a, [3, 2, 0, 1, 4])
    panel_b = _trim_white_border(_render_pdf(FIG06_PANEL_B, scale=4.0), padding=4)
    panel_c = _trim_white_border(_render_pdf(FIG06_PANEL_C, scale=4.0), padding=4)
    panel_d = _trim_white_border(_render_pdf(FIG06_PANEL_D, scale=4.0), padding=4)
    panel_e = _trim_white_border(_render_pdf(FIG06_PANEL_E, scale=4.0), padding=4)
    figure_width_px = _mm_to_px(FIG06_WIDTH_MM)
    figure_height_px = _mm_to_px(FIG06_HEIGHT_MM)
    outer_margin_x_px = _mm_to_px(FIG06_OUTER_MARGIN_X_MM)
    outer_margin_top_px = _mm_to_px(FIG06_OUTER_MARGIN_TOP_MM)
    outer_margin_bottom_px = _mm_to_px(FIG06_OUTER_MARGIN_BOTTOM_MM)
    a_height_px = _mm_to_px(FIG06_A_HEIGHT_MM)
    b_height_px = _mm_to_px(FIG06_B_HEIGHT_MM)
    c_height_px = _mm_to_px(FIG06_C_HEIGHT_MM)
    d_height_px = _mm_to_px(FIG06_D_HEIGHT_MM)
    panel_c_width_px = _mm_to_px(FIG06_PANEL_C_WIDTH_MM)
    col_gap_px = _mm_to_px(FIG06_COL_GAP_MM)
    row_gap_px = _mm_to_px(FIG06_ROW_GAP_MM)
    inner_width_px = figure_width_px - 2 * outer_margin_x_px
    panel_e_width_px = inner_width_px - panel_c_width_px - col_gap_px
    if panel_e_width_px <= 0:
        raise RuntimeError(
            "Fig. 6 compose contract is inconsistent after px rounding: "
            f"{inner_width_px=}, {panel_c_width_px=}, {panel_e_width_px=}, {col_gap_px=}"
        )

    panel_a = _contain_in_box(panel_a, inner_width_px, a_height_px)
    panel_b = _contain_in_box(panel_b, inner_width_px, b_height_px)
    panel_c = _contain_in_box(panel_c, panel_c_width_px, c_height_px)
    panel_d = _contain_in_box(panel_d, panel_e_width_px, c_height_px)
    panel_e = _contain_in_box(panel_e, inner_width_px, d_height_px)
    _save_composite(panel_a, panel_a_manuscript_asset)

    canvas = Image.new("RGB", (figure_width_px, figure_height_px), "white")
    draw = ImageDraw.Draw(canvas)
    row_a_y = outer_margin_top_px
    row_b_y = row_a_y + a_height_px + row_gap_px
    row_ce_y = row_b_y + b_height_px + row_gap_px
    row_d_y = row_ce_y + c_height_px + row_gap_px
    panel_a_x = outer_margin_x_px
    panel_b_x = outer_margin_x_px
    panel_c_x = outer_margin_x_px
    panel_e_x = panel_c_x + panel_c_width_px + col_gap_px
    panel_d_x = outer_margin_x_px

    canvas.paste(panel_a, (panel_a_x, row_a_y))
    canvas.paste(panel_b, (panel_b_x, row_b_y))
    canvas.paste(panel_c, (panel_c_x, row_ce_y))
    canvas.paste(panel_d, (panel_e_x, row_ce_y))
    canvas.paste(panel_e, (panel_d_x, row_d_y))

    _draw_panel_label(
        draw,
        "a",
        panel_a_x + _mm_to_px(2.0),
        row_a_y + _mm_to_px(1.5),
    )

    _save_composite(canvas, fig06_asset)
    manuscript_manifest = manuscript_panel_dir / "fig06_panel_manifest.json"
    _write_reference_panel_manifest(
        manuscript_manifest,
        figure_id="fig06",
        composite_asset="paper/figures/fig06_universality.jpg",
        panels=[
            {
                "panel_id": "a",
                "title": "Material exemplars",
                "asset_path": "figures/output/fig06_universality_manuscript_panels/fig06_panel_a_material_exemplars.png",
                "provenance_mode": "manual_support",
            },
            {
                "panel_id": "b",
                "title": "Cross-material H",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_b_cross_material_h.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "c",
                "title": "Low-rank continuity",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_c_low_rank_continuity.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "d",
                "title": "Screening consequence",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_d_screening_consequence.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "e",
                "title": "Material frequency structure",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_e_material_frequency_structure.pdf",
                "provenance_mode": "data_backed",
            },
        ],
    )
    _write_layout_metadata(
        fig06_layout_asset,
        {
            "contract_version": contract_version(),
            "figure_mm": {"width": FIG06_WIDTH_MM, "height": FIG06_HEIGHT_MM},
            "axes": [
                {
                    "index": 0,
                    "panel_id": "a",
                    "kind": "manual",
                    "has_data": False,
                    "title": "Material exemplars",
                    **_bbox_payload(
                        FIG06_OUTER_MARGIN_X_MM,
                        FIG06_HEIGHT_MM - FIG06_OUTER_MARGIN_TOP_MM - FIG06_A_HEIGHT_MM,
                        FIG06_WIDTH_MM - 2 * FIG06_OUTER_MARGIN_X_MM,
                        FIG06_A_HEIGHT_MM,
                        FIG06_WIDTH_MM, FIG06_HEIGHT_MM,
                    ),
                },
                {
                    "index": 1,
                    "panel_id": "b",
                    "kind": "rectilinear",
                    "has_data": True,
                    "title": "Cross-material H",
                    **_bbox_payload(
                        FIG06_OUTER_MARGIN_X_MM,
                        FIG06_OUTER_MARGIN_BOTTOM_MM + FIG06_D_HEIGHT_MM + FIG06_C_HEIGHT_MM + 2 * FIG06_ROW_GAP_MM,
                        FIG06_WIDTH_MM - 2 * FIG06_OUTER_MARGIN_X_MM,
                        FIG06_B_HEIGHT_MM,
                        FIG06_WIDTH_MM, FIG06_HEIGHT_MM,
                    ),
                },
                {
                    "index": 2,
                    "panel_id": "c",
                    "kind": "rectilinear",
                    "has_data": True,
                    "title": "Low-rank continuity",
                    **_bbox_payload(
                        FIG06_OUTER_MARGIN_X_MM,
                        FIG06_OUTER_MARGIN_BOTTOM_MM + FIG06_D_HEIGHT_MM + FIG06_ROW_GAP_MM,
                        FIG06_PANEL_C_WIDTH_MM,
                        FIG06_C_HEIGHT_MM,
                        FIG06_WIDTH_MM, FIG06_HEIGHT_MM,
                    ),
                },
                {
                    "index": 3,
                    "panel_id": "d",
                    "kind": "rectilinear",
                    "has_data": True,
                    "title": "Screening consequence",
                    **_bbox_payload(
                        FIG06_OUTER_MARGIN_X_MM + FIG06_PANEL_C_WIDTH_MM + FIG06_COL_GAP_MM,
                        FIG06_OUTER_MARGIN_BOTTOM_MM + FIG06_D_HEIGHT_MM + FIG06_ROW_GAP_MM,
                        FIG06_PANEL_E_WIDTH_MM,
                        FIG06_C_HEIGHT_MM,
                        FIG06_WIDTH_MM, FIG06_HEIGHT_MM,
                    ),
                },
                {
                    "index": 4,
                    "panel_id": "e",
                    "kind": "rectilinear",
                    "has_data": True,
                    "title": "Material frequency structure",
                    **_bbox_payload(
                        FIG06_OUTER_MARGIN_X_MM,
                        FIG06_OUTER_MARGIN_BOTTOM_MM,
                        FIG06_WIDTH_MM - 2 * FIG06_OUTER_MARGIN_X_MM,
                        FIG06_D_HEIGHT_MM,
                        FIG06_WIDTH_MM, FIG06_HEIGHT_MM,
                    ),
                },
            ],
            "source_layout_spec": source_layout_spec(),
            "typography_pt": TYPOGRAPHY_PT,
        },
    )
    return [fig06_asset, fig06_layout_asset, manuscript_manifest]


COMPOSERS = {
    "fig01": compose_fig01,
    "fig02": compose_fig02,
    "fig03": compose_fig03,
    "fig04": compose_fig04,
    "fig05": compose_fig05,
    "fig06": compose_fig06,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose active manuscript figures from panel and generator assets.")
    parser.add_argument(
        "--paper-dir",
        default=str(DEFAULT_PAPER_DIR),
        help="Target directory for composed manuscript assets.",
    )
    parser.add_argument(
        "--figures",
        help="Comma-separated active figure ids to compose (default: fig01,fig02,fig03,fig04,fig05,fig06).",
    )
    args = parser.parse_args()
    paper_dir = Path(args.paper_dir)
    selected_ids = _normalize_figure_ids(args.figures)

    created: list[Path] = []
    for figure_id in selected_ids:
        created.extend(COMPOSERS[figure_id](paper_dir))
    print("\n".join(str(path) for path in created))


if __name__ == "__main__":
    main()
