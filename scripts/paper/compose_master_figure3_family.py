#!/usr/bin/env python3
"""Compose panel-first paper-facing figures (6-figure manuscript set).

This script rebuilds the manuscript assets for the current main-paper figure set
from generator composite assets whenever possible:

- Fig. 1: Paradigm Shift (5 panels: a fixed support + b,c,d,e data-backed)
- Fig. 2: SVD Spectrum (7 panels: all generated as composite PDF)
- Fig. 3: Speech-side compactness + neighborhood broadening (6 panels: all generated)
- Fig. 4: Neighborhood-preserving admissibility (5 panels: all generated)
- Fig. 5: Performance + Structure (5 panels: all generated)
- Fig. 6: Universality (5 panels: all generated under governed layout)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

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
    figure_typography,
    font_pt,
    font_tokens,
    stroke_pt,
    pt_to_px,
    source_layout_spec,
)
from figures.style import SEMANTIC_PALETTE, STYLE_COLORS


DEFAULT_PAPER_DIR = REPO_ROOT / "paper/figures"
MM_PER_INCH = 25.4
COMPOSE_DPI = 300
PANEL_LABEL_PT = font_pt("panel_label")
TYPOGRAPHY_PT = font_tokens()
ACTIVE_FIGURE_IDS = ("fig01", "fig02", "fig03", "fig04", "fig05", "fig06")

# --- Figure 1: Paradigm Shift (5 panels: a fixed support + b support schematic + c,d,e data-backed) ---
FIG01_PANEL_A = REPO_ROOT / "figures/output/fig01_paradigm_shift_panels/fig01_panel_a_experimental_setup.png"
FIG01_PANEL_B = REPO_ROOT / "figures/output/fig01_paradigm_shift_panels/fig01_panel_b_spectral_fingerprint.png"
FIG01_COMPOSITE_CDE = REPO_ROOT / "figures/output/fig01_paradigm_data.pdf"
FIG01_COMPOSITE_CDE_LAYOUT = REPO_ROOT / "figures/output/fig01_paradigm_data.layout.json"
FIG01_COMPOSE = figure_section("fig01", "compose")
FIG01_WIDTH_MM = float(FIG01_COMPOSE["width_mm"])
FIG01_PANEL_A_WIDTH_MM = float(FIG01_COMPOSE["panel_a_width_mm"])
FIG01_PANEL_B_WIDTH_MM = float(FIG01_COMPOSE["panel_b_width_mm"])
FIG01_ROW_HEIGHT_MM = float(FIG01_COMPOSE["row_height_mm"])
FIG01_BOTTOM_ROW_HEIGHT_MM = float(FIG01_COMPOSE.get("bottom_row_height_mm", FIG01_ROW_HEIGHT_MM))
FIG01_ROW_GAP_MM = float(FIG01_COMPOSE["row_gap_mm"])
FIG01_HEIGHT_MM = float(FIG01_COMPOSE["height_mm"])
FIG01_PANEL_A_CROP = (0.00, 0.12, 0.60, 0.78)
FIG01_PANEL_B_CROP = (0.00, 0.00, 1.00, 1.00)
FIG01_TYPOGRAPHY_PT = figure_typography("fig01")

# --- Figure 2: SVD Spectrum (7 panels: all generated as composite PDF) ---
FIG02_COMPOSITE = REPO_ROOT / "figures/output/fig02_svd_spectrum.pdf"
FIG02_COMPOSITE_LAYOUT = FIG02_COMPOSITE.with_suffix(".layout.json")
FIG02_COMPOSE = figure_section("fig02", "compose")
FIG02_WIDTH_MM = float(FIG02_COMPOSE["width_mm"])
FIG02_HEIGHT_MM = float(FIG02_COMPOSE["height_mm"])

# --- Figure 3: Fingerprint Discriminability (5 panels: all generated) ---
FIG03_COMPOSITE = REPO_ROOT / "figures/output/fig03_fingerprint_discriminability.pdf"
FIG03_COMPOSITE_LAYOUT = FIG03_COMPOSITE.with_suffix(".layout.json")
FIG03_COMPOSE = figure_section("fig03", "compose")
FIG03_WIDTH_MM = float(FIG03_COMPOSE["width_mm"])
FIG03_HEIGHT_MM = float(FIG03_COMPOSE["height_mm"])

# --- Figure 4: Solver Mechanism (5 panels: all generated as composite PDF) ---
FIG04_COMPOSITE = REPO_ROOT / "figures/output/fig04_solver_dynamics.pdf"
FIG04_COMPOSITE_LAYOUT = FIG04_COMPOSITE.with_suffix(".layout.json")
FIG04_COMPOSE = figure_section("fig04", "compose")
FIG04_WIDTH_MM = float(FIG04_COMPOSE["width_mm"])
FIG04_HEIGHT_MM = float(FIG04_COMPOSE["height_mm"])
# --- Figure 5: Performance + Structure (5 panels: all generated) ---
FIG05_COMPOSITE = REPO_ROOT / "figures/output/fig05_performance_structure.pdf"
FIG05_COMPOSITE_LAYOUT = FIG05_COMPOSITE.with_suffix(".layout.json")
FIG05_COMPOSE = figure_section("fig05", "compose")
FIG05_WIDTH_MM = float(FIG05_COMPOSE["width_mm"])
FIG05_HEIGHT_MM = float(FIG05_COMPOSE["height_mm"])

# --- Figure 6: Universality (5 panels: all generated under governed layout) ---
FIG06_COMPOSITE = REPO_ROOT / "figures/output/fig06_universality.pdf"
FIG06_COMPOSITE_LAYOUT = FIG06_COMPOSITE.with_suffix(".layout.json")
FIG06_PANEL_A = REPO_ROOT / "figures/output/fig06_universality_panels/fig06_panel_a_response_regime.pdf"
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
    FIG06_ROW_CD_HEIGHT_MM = float(FIG06_COMPOSE["row_cd_height_mm"])
    FIG06_PANEL_D_WIDTH_MM = float(FIG06_COMPOSE["panel_d_width_mm"])
    FIG06_PANEL_E_HEIGHT_MM = float(FIG06_COMPOSE["panel_e_height_mm"])
    FIG06_COL_GAP_MM = float(FIG06_COMPOSE["col_gap_mm"])
    FIG06_ROW_GAP_MM = float(FIG06_COMPOSE["row_gap_mm"])
except Exception as exc:  # pragma: no cover - only triggered by unrelated fig06 contract drift
    FIG06_LAYOUT_ERROR = exc
    FIG06_COMPOSE = {}
    FIG06_WIDTH_MM = FIG06_HEIGHT_MM = 0.0
    FIG06_OUTER_MARGIN_X_MM = FIG06_OUTER_MARGIN_TOP_MM = FIG06_OUTER_MARGIN_BOTTOM_MM = 0.0
    FIG06_A_HEIGHT_MM = FIG06_B_HEIGHT_MM = FIG06_ROW_CD_HEIGHT_MM = FIG06_PANEL_E_HEIGHT_MM = 0.0
    FIG06_PANEL_D_WIDTH_MM = FIG06_COL_GAP_MM = 0.0
    FIG06_ROW_GAP_MM = 0.0
FIG06_TYPOGRAPHY_PT = figure_typography("fig06")


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


def _load_regular_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
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


def _crop_relative(
    img: Image.Image,
    left_frac: float,
    top_frac: float,
    right_frac: float,
    bottom_frac: float,
) -> Image.Image:
    """Crop an image using normalized fractions of its current width and height."""
    width, height = img.size
    left = int(round(width * left_frac))
    top = int(round(height * top_frac))
    right = int(round(width * right_frac))
    bottom = int(round(height * bottom_frac))
    return img.crop((left, top, right, bottom))


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


def _render_pdf_to_size(pdf_path: Path, width_px: int, height_px: int) -> Image.Image:
    rendered = _render_pdf(pdf_path, scale=4.0).convert("RGB")
    return rendered.resize((width_px, height_px), Image.Resampling.LANCZOS)


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


def _mm_to_px(mm: float) -> int:
    return int(round(mm / MM_PER_INCH * COMPOSE_DPI))


def _stroke_px(token_name: str, dpi: int, *, minimum: int = 1) -> int:
    return max(minimum, pt_to_px(stroke_pt(token_name), dpi))


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
        "contract_version": contract_version(),
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
        "source_layout_spec": source_layout_spec(),
        "storage_mode": "reference_existing_outputs",
        "typography_pt": figure_typography(figure_id),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _layout_payload_with_figure_mm(
    payload: dict[str, Any],
    *,
    figure_width_mm: float,
    figure_height_mm: float,
) -> dict[str, Any]:
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
            bbox_norm = axis.get(norm_key)
            if bbox_norm is None:
                entry[mm_key] = None
                continue
            entry[mm_key] = {
                "x0": round(float(bbox_norm["x0"]) * figure_width_mm, 3),
                "y0": round(float(bbox_norm["y0"]) * figure_height_mm, 3),
                "width": round(float(bbox_norm["width"]) * figure_width_mm, 3),
                "height": round(float(bbox_norm["height"]) * figure_height_mm, 3),
            }
        axes_payload.append(entry)

    return {
        "contract_version": payload.get("contract_version", contract_version()),
        "figure_mm": {
            "width": round(float(figure_width_mm), 3),
            "height": round(float(figure_height_mm), 3),
        },
        "axes": axes_payload,
        "source_layout_spec": payload.get("source_layout_spec", source_layout_spec()),
        "typography_pt": payload.get("typography_pt", TYPOGRAPHY_PT),
    }


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


def _mask_rectangles_with_overlay(
    img: Image.Image,
    rectangles: list[tuple[int, int, int, int]],
    *,
    fill: tuple[int, int, int, int],
    radius: int = 10,
) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for rect in rectangles:
        draw.rounded_rectangle(rect, radius=radius, fill=fill)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _annotate_fig01_setup_panel(img: Image.Image) -> Image.Image:
    photo = img.convert("RGB")
    texture_patches = [
        {"src": (366, 110, 664, 198), "dst": (382, 356, 728, 494)},
    ]
    for patch in texture_patches:
        src = photo.crop(patch["src"])
        dst = patch["dst"]
        src = src.resize((dst[2] - dst[0], dst[3] - dst[1]), Image.Resampling.BICUBIC)
        photo.paste(src, dst)
    tag_specs = {
        "Speaker": {"rect": (74, 266, 320, 392), "alpha": 168},
        "Acrylic plate": {"rect": (398, 260, 724, 420), "alpha": 182},
    }
    for spec in tag_specs.values():
        rect = spec["rect"]
        region = photo.crop(rect).filter(ImageFilter.GaussianBlur(radius=9))
        photo.paste(region, rect)
    for spec in tag_specs.values():
        photo = _mask_rectangles_with_overlay(
            photo,
            [spec["rect"]],
            fill=(18, 18, 18, spec["alpha"]),
        )

    draw = ImageDraw.Draw(photo)
    label_font = _load_regular_font(pt_to_px(FIG01_TYPOGRAPHY_PT["tick_label"], COMPOSE_DPI))
    line_color = STYLE_COLORS["guide_line"]
    text_color = (255, 255, 255)
    shadow_color = (12, 12, 12)
    arrow_width = _stroke_px("annotation", COMPOSE_DPI)

    for label, spec in tag_specs.items():
        rect = spec["rect"]
        text_bbox = draw.textbbox((0, 0), label, font=label_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        anchor_x = rect[0] + (rect[2] - rect[0] - text_width) // 2
        anchor_y = rect[1] + (rect[3] - rect[1] - text_height) // 2 - 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            draw.text((anchor_x + dx, anchor_y + dy), label, fill=shadow_color, font=label_font)
        draw.text((anchor_x, anchor_y), label, fill=text_color, font=label_font)

    ldv_anchor = (548, 166)
    ldv_target = (455, 382)
    draw.line([ldv_anchor, ldv_target], fill=line_color, width=arrow_width)
    draw.line([ldv_target, (ldv_target[0] - 9, ldv_target[1] - 8)], fill=line_color, width=arrow_width)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        draw.text((ldv_anchor[0] + dx, ldv_anchor[1] + dy), "LDV spot", fill=shadow_color, font=label_font)
    draw.text(ldv_anchor, "LDV spot", fill=text_color, font=label_font)
    return photo


def _clear_fig01_panel_b_header(img: Image.Image) -> Image.Image:
    """Remove the presentation-style headline while keeping the legacy schematic body."""
    panel = ImageOps.exif_transpose(img).convert("RGB")
    draw = ImageDraw.Draw(panel)
    cutoff_y = int(round(panel.height * 0.12))
    draw.rectangle((0, 0, panel.width, cutoff_y), fill="white")
    return panel


def compose_fig01(paper_dir: Path) -> list[Path]:
    """Fig 1: mm-driven layout with fixed a,b panels and the generated c,d,e strip."""
    fig01_asset = paper_dir / "fig01_paradigm-shift.jpg"
    fig01_layout_asset = fig01_asset.with_suffix(".layout.json")
    manuscript_manifest = REPO_ROOT / "figures/output/fig01_paradigm_shift_panels/fig01_panel_manifest.json"

    panel_a = ImageOps.exif_transpose(Image.open(FIG01_PANEL_A)).convert("RGB")
    panel_b = ImageOps.exif_transpose(Image.open(FIG01_PANEL_B)).convert("RGB")
    panel_a = _crop_relative(panel_a, *FIG01_PANEL_A_CROP)
    panel_b = _clear_fig01_panel_b_header(panel_b)
    panel_b = _trim_white_border(panel_b, padding=0)
    panel_b = _crop_relative(panel_b, *FIG01_PANEL_B_CROP)
    panel_cde = _render_pdf(FIG01_COMPOSITE_CDE, scale=4.0).convert("RGB")

    figure_width_px = _mm_to_px(FIG01_WIDTH_MM)
    panel_a_width_px = _mm_to_px(FIG01_PANEL_A_WIDTH_MM)
    panel_b_width_px = _mm_to_px(FIG01_PANEL_B_WIDTH_MM)
    top_gap_px = figure_width_px - panel_a_width_px - panel_b_width_px
    top_row_height_px = _mm_to_px(FIG01_ROW_HEIGHT_MM)
    bottom_row_height_px = _mm_to_px(FIG01_BOTTOM_ROW_HEIGHT_MM)
    row_gap_px = _mm_to_px(FIG01_ROW_GAP_MM)
    figure_height_px = _mm_to_px(FIG01_HEIGHT_MM)

    panel_a = _contain_in_box(panel_a, panel_a_width_px, top_row_height_px)
    panel_a = _annotate_fig01_setup_panel(panel_a)
    panel_b = _contain_in_box(panel_b, panel_b_width_px, top_row_height_px)
    panel_cde = _resize_to_width(panel_cde, figure_width_px)

    canvas = Image.new("RGB", (figure_width_px, figure_height_px), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(panel_a, (0, 0))
    canvas.paste(panel_b, (panel_a_width_px + top_gap_px, 0))
    _draw_panel_label(draw, "a", _mm_to_px(2.5), _mm_to_px(2.0))
    _draw_panel_label(draw, "b", panel_a_width_px + top_gap_px + _mm_to_px(2.5), _mm_to_px(2.0))

    bottom_row_y_px = top_row_height_px + row_gap_px
    bottom_strip_y_px = bottom_row_y_px + max((bottom_row_height_px - panel_cde.height) // 2, 0)
    canvas.paste(panel_cde, (0, bottom_strip_y_px))

    _save_composite(canvas, fig01_asset)

    composite_layout = json.loads(FIG01_COMPOSITE_CDE_LAYOUT.read_text(encoding="utf-8"))
    bottom_scale = FIG01_WIDTH_MM / composite_layout["figure_mm"]["width"]
    bottom_strip_height_mm = composite_layout["figure_mm"]["height"] * bottom_scale
    bottom_row_offset_mm = max((FIG01_BOTTOM_ROW_HEIGHT_MM - bottom_strip_height_mm) / 2.0, 0.0)

    axes = [
        {
            "index": 0,
            "panel_id": "a",
            "kind": "manual",
            "has_data": False,
            "title": "Experimental setup",
            **_bbox_payload(
                x0_mm=0.0,
                y0_mm=FIG01_BOTTOM_ROW_HEIGHT_MM + FIG01_ROW_GAP_MM,
                width_mm=FIG01_PANEL_A_WIDTH_MM,
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
            "title": "SM1 physical-principle schematic",
            **_bbox_payload(
                x0_mm=FIG01_PANEL_A_WIDTH_MM + (FIG01_WIDTH_MM - FIG01_PANEL_A_WIDTH_MM - FIG01_PANEL_B_WIDTH_MM),
                y0_mm=FIG01_BOTTOM_ROW_HEIGHT_MM + FIG01_ROW_GAP_MM,
                width_mm=FIG01_PANEL_B_WIDTH_MM,
                height_mm=FIG01_ROW_HEIGHT_MM,
                figure_width_mm=FIG01_WIDTH_MM,
                figure_height_mm=FIG01_HEIGHT_MM,
            ),
        },
    ]

    panel_ids = ("c", "d", "e")
    data_axes = [ax for ax in composite_layout.get("axes", []) if ax.get("title")]
    if len(data_axes) < len(panel_ids):
        raise ValueError(
            f"Expected at least {len(panel_ids)} titled data axes in Fig. 1 layout, got {len(data_axes)}"
        )
    for ax_meta, panel_id in zip(data_axes[: len(panel_ids)], panel_ids, strict=True):
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
            "typography_pt": FIG01_TYPOGRAPHY_PT,
        },
    )
    _write_reference_panel_manifest(
        manuscript_manifest,
        figure_id="fig01",
        composite_asset="paper/figures/fig01_paradigm-shift.jpg",
        panels=[
            {
                "panel_id": "a",
                "title": "Experimental setup",
                "asset_path": "figures/output/fig01_paradigm_shift_panels/fig01_panel_a_experimental_setup.png",
                "provenance_mode": "manual_support",
            },
            {
                "panel_id": "b",
                "title": "SM1 physical-principle schematic",
                "asset_path": "figures/output/fig01_paradigm_shift_panels/fig01_panel_b_spectral_fingerprint.png",
                "provenance_mode": "manual_support",
            },
            {
                "panel_id": "c",
                "title": "Spectral shaping with repeatability",
                "asset_path": "figures/output/fig01_paradigm_data_panels/fig01_panel_c_spectral_shaping_repeatability.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "d",
                "title": "Frequency-dependent directivity",
                "asset_path": "figures/output/fig01_paradigm_data_panels/fig01_panel_d_directivity.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "e",
                "title": "Inter-angle fingerprint similarity",
                "asset_path": "figures/output/fig01_paradigm_data_panels/fig01_panel_e_similarity.pdf",
                "provenance_mode": "data_backed",
            },
        ],
    )
    return [fig01_asset, fig01_layout_asset, manuscript_manifest]


def compose_fig02(paper_dir: Path) -> list[Path]:
    """Fig 2: All 6 panels from composite PDF (a-f)."""
    fig02_asset = paper_dir / "fig02_svd-physical-dictionary.jpg"
    fig02_layout_asset = fig02_asset.with_suffix(".layout.json")

    composite = _render_pdf_to_size(
        FIG02_COMPOSITE,
        _mm_to_px(FIG02_WIDTH_MM),
        _mm_to_px(FIG02_HEIGHT_MM),
    )
    _save_composite(composite, fig02_asset)
    if not FIG02_COMPOSITE_LAYOUT.exists():
        raise FileNotFoundError(f"Missing Fig. 2 composite layout sidecar: {FIG02_COMPOSITE_LAYOUT}")
    composite_layout = json.loads(FIG02_COMPOSITE_LAYOUT.read_text(encoding="utf-8"))
    manuscript_layout = _layout_payload_with_figure_mm(
        composite_layout,
        figure_width_mm=FIG02_WIDTH_MM,
        figure_height_mm=FIG02_HEIGHT_MM,
    )
    _write_layout_metadata(fig02_layout_asset, manuscript_layout)
    return [fig02_asset, fig02_layout_asset]


def compose_fig03(paper_dir: Path) -> list[Path]:
    """Fig. 3: all 6 generated panels promoted from the composite PDF."""
    fig03_asset = paper_dir / "fig03_fingerprint-discriminability.jpg"
    fig03_layout_asset = fig03_asset.with_suffix(".layout.json")

    composite = _render_pdf_to_size(
        FIG03_COMPOSITE,
        _mm_to_px(FIG03_WIDTH_MM),
        _mm_to_px(FIG03_HEIGHT_MM),
    )
    _save_composite(composite, fig03_asset)
    if not FIG03_COMPOSITE_LAYOUT.exists():
        raise FileNotFoundError(f"Missing Fig. 3 composite layout sidecar: {FIG03_COMPOSITE_LAYOUT}")
    composite_layout = json.loads(FIG03_COMPOSITE_LAYOUT.read_text(encoding="utf-8"))
    manuscript_layout = _layout_payload_with_figure_mm(
        composite_layout,
        figure_width_mm=FIG03_WIDTH_MM,
        figure_height_mm=FIG03_HEIGHT_MM,
    )
    _write_layout_metadata(fig03_layout_asset, manuscript_layout)
    return [fig03_asset, fig03_layout_asset]


def compose_fig04(paper_dir: Path) -> list[Path]:
    """Fig. 4: governed generated composite promoted to the paper-facing JPG asset."""
    fig04_asset = paper_dir / "fig04_solver-dynamics.jpg"
    fig04_layout_asset = fig04_asset.with_suffix(".layout.json")
    manuscript_panel_dir = REPO_ROOT / "figures/output/fig04_solver_dynamics_manuscript_panels"
    if not FIG04_COMPOSITE.exists():
        raise FileNotFoundError(f"Missing Fig. 4 composite asset: {FIG04_COMPOSITE}")
    rendered = _render_pdf_to_size(
        FIG04_COMPOSITE,
        _mm_to_px(FIG04_WIDTH_MM),
        _mm_to_px(FIG04_HEIGHT_MM),
    )
    _save_composite(rendered, fig04_asset)
    manuscript_manifest = manuscript_panel_dir / "fig04_panel_manifest.json"
    _write_reference_panel_manifest(
        manuscript_manifest,
        figure_id="fig04",
        composite_asset="paper/figures/fig04_solver-dynamics.jpg",
        panels=[
            {
                "panel_id": "a",
                "title": "Neighborhood admissibility strip",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_a_architecture_physics.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "b",
                "title": "Broad initial match",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_b_broad_match.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "c",
                "title": "Local contraction",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_c_local_gate.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "d",
                "title": "Validation-wide neighborhood contraction",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_d_residual_purification.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "e",
                "title": "Within-15° mass gain",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_e_ablation.pdf",
                "provenance_mode": "data_backed",
            },
        ],
    )
    if not FIG04_COMPOSITE_LAYOUT.exists():
        raise FileNotFoundError(f"Missing Fig. 4 composite layout sidecar: {FIG04_COMPOSITE_LAYOUT}")
    composite_layout = json.loads(FIG04_COMPOSITE_LAYOUT.read_text(encoding="utf-8"))
    _write_layout_metadata(
        fig04_layout_asset,
        _layout_payload_with_figure_mm(
            composite_layout,
            figure_width_mm=FIG04_WIDTH_MM,
            figure_height_mm=FIG04_HEIGHT_MM,
        ),
    )
    return [fig04_asset, fig04_layout_asset, manuscript_manifest]


def compose_fig05(paper_dir: Path) -> list[Path]:
    """Fig 5: All 5 panels from composite PDF (a-e)."""
    fig05_asset = paper_dir / "fig05_performance-structure.jpg"
    fig05_layout_asset = fig05_asset.with_suffix(".layout.json")

    composite = _render_pdf_to_size(
        FIG05_COMPOSITE,
        _mm_to_px(FIG05_WIDTH_MM),
        _mm_to_px(FIG05_HEIGHT_MM),
    )
    _save_composite(composite, fig05_asset)
    if FIG05_COMPOSITE_LAYOUT.exists():
        composite_layout = json.loads(FIG05_COMPOSITE_LAYOUT.read_text(encoding="utf-8"))
        manuscript_layout = _layout_payload_with_figure_mm(
            composite_layout,
            figure_width_mm=FIG05_WIDTH_MM,
            figure_height_mm=FIG05_HEIGHT_MM,
        )
        _write_layout_metadata(fig05_layout_asset, manuscript_layout)
        return [fig05_asset, fig05_layout_asset]
    return [fig05_asset]


def compose_fig06(paper_dir: Path) -> list[Path]:
    """Fig 6: governed generated composite promoted to the paper-facing JPG asset."""
    if FIG06_LAYOUT_ERROR is not None:
        raise RuntimeError(
            "Fig. 6 compose contract is invalid in the current worktree; "
            "fix the fig06 layout contract before composing fig06"
        ) from FIG06_LAYOUT_ERROR

    fig06_asset = paper_dir / "fig06_universality.jpg"
    fig06_layout_asset = fig06_asset.with_suffix(".layout.json")
    manuscript_panel_dir = REPO_ROOT / "figures/output/fig06_universality_manuscript_panels"
    if not FIG06_COMPOSITE.exists():
        raise FileNotFoundError(f"Missing Fig. 6 composite asset: {FIG06_COMPOSITE}")
    rendered = _render_pdf_to_size(
        FIG06_COMPOSITE,
        _mm_to_px(FIG06_WIDTH_MM),
        _mm_to_px(FIG06_HEIGHT_MM),
    )
    _save_composite(rendered, fig06_asset)
    manuscript_manifest = manuscript_panel_dir / "fig06_panel_manifest.json"
    _write_reference_panel_manifest(
        manuscript_manifest,
        figure_id="fig06",
        composite_asset="paper/figures/fig06_universality.jpg",
        panels=[
            {
                "panel_id": "a",
                "title": "Measured response regime",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_a_response_regime.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "b",
                "title": "Cross-material H",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_b_cross_material_h.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "c",
                "title": "Local-ordering decay",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_c_low_rank_continuity.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "d",
                "title": "Readout versus overlap burden",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_d_screening_consequence.pdf",
                "provenance_mode": "data_backed",
            },
            {
                "panel_id": "e",
                "title": "Object-conditioned contrast band",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_e_material_frequency_structure.pdf",
                "provenance_mode": "data_backed",
            },
        ],
    )
    if not FIG06_COMPOSITE_LAYOUT.exists():
        raise FileNotFoundError(f"Missing Fig. 6 composite layout sidecar: {FIG06_COMPOSITE_LAYOUT}")
    composite_layout = json.loads(FIG06_COMPOSITE_LAYOUT.read_text(encoding="utf-8"))
    _write_layout_metadata(
        fig06_layout_asset,
        _layout_payload_with_figure_mm(
            composite_layout,
            figure_width_mm=FIG06_WIDTH_MM,
            figure_height_mm=FIG06_HEIGHT_MM,
        ),
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
