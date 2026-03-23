#!/usr/bin/env python3
"""Compose panel-first paper-facing figures (6-figure, 32-panel structure).

This script rebuilds the manuscript assets for the current main-paper figure set
from panel-level assets whenever possible:

- Fig. 1: Paradigm Shift (5 panels: a,b external + c,d,e generated)
- Fig. 2: SVD Spectrum (7 panels: all generated as composite PDF)
- Fig. 3: Fingerprint Discriminability (5 panels: all generated)
- Fig. 4: Solver Dynamics (4 panels: a external + b,c,d generated)
- Fig. 5: Performance + Structure (6 panels: all generated)
- Fig. 6: Universality (5 panels: a,b,c external + d,e generated)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

try:
    import fitz
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("PyMuPDF is required for figure composition") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
MM_PER_INCH = 25.4
COMPOSE_DPI = 300

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

# --- Figure 2: SVD Spectrum (7 panels: all generated as composite PDF) ---
FIG02_COMPOSITE = REPO_ROOT / "figures/output/fig02_svd_spectrum.pdf"

# --- Figure 3: Fingerprint Discriminability (5 panels: all generated) ---
FIG03_COMPOSITE = REPO_ROOT / "figures/output/fig03_fingerprint_discriminability.pdf"

# --- Figure 4: Solver Dynamics (4 panels: a external + b,c,d generated) ---
FIG04_PANEL_A = REPO_ROOT / "paper/figures/fig04_unrolled-attention-omp.jpg"
FIG04_COMPOSITE_BCD = REPO_ROOT / "figures/output/fig04_solver_dynamics.pdf"

# --- Figure 5: Performance + Structure (6 panels: all generated) ---
FIG05_COMPOSITE = REPO_ROOT / "figures/output/fig05_performance_structure.pdf"

# --- Figure 6: Universality (5 panels: a,b,c external + d,e generated) ---
FIG06_PANELS_ABC_DIR = REPO_ROOT / "figures/output/fig06_cross_material_universality_panels"
FIG06_COMPOSITE_DE = REPO_ROOT / "figures/output/fig06_universality.pdf"


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


def _trim_white_border(img: Image.Image, padding: int = 8) -> Image.Image:
    rgb = ImageOps.exif_transpose(img).convert("RGB")
    bg = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return rgb
    left = max(bbox[0] - padding, 0)
    top = max(bbox[1] - padding, 0)
    right = min(bbox[2] + padding, rgb.width)
    bottom = min(bbox[3] + padding, rgb.height)
    return rgb.crop((left, top, right, bottom))


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
    scale = min(box_width / max(img.width, 1), box_height / max(img.height, 1))
    target_width = int(round(img.width * scale))
    target_height = int(round(img.height * scale))
    resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (box_width, box_height), "white")
    x = (box_width - target_width) // 2
    y = (box_height - target_height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _mm_to_px(mm: float) -> int:
    return int(round(mm / MM_PER_INCH * COMPOSE_DPI))


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


def _save_composite(img: Image.Image, path: Path) -> None:
    _ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        img.save(path, format="JPEG", quality=95, subsampling=0, dpi=(COMPOSE_DPI, COMPOSE_DPI))
    elif suffix == ".png":
        img.save(path, format="PNG", dpi=(COMPOSE_DPI, COMPOSE_DPI))
    else:
        raise ValueError(f"Unsupported composite suffix for {path}")


def compose_fig01() -> list[Path]:
    """Fig 1: mm-driven layout with fixed a,b panels and the generated c,d,e strip."""
    fig01_asset = REPO_ROOT / "paper/figures/fig01_paradigm-shift.jpg"
    fig01_layout_asset = fig01_asset.with_suffix(".layout.json")

    panel_a = ImageOps.exif_transpose(Image.open(FIG01_PANEL_A)).convert("RGB")
    panel_b = ImageOps.exif_transpose(Image.open(FIG01_PANEL_B)).convert("RGB")
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
    canvas.paste(panel_a, (0, 0))
    canvas.paste(panel_b, (top_panel_width_px + top_gap_px, 0))

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
            "figure_mm": {
                "width": FIG01_WIDTH_MM,
                "height": FIG01_HEIGHT_MM,
            },
            "axes": axes,
            "source_layout_spec": "figures/conf/layout_spec.md",
        },
    )
    return [fig01_asset, fig01_layout_asset]


def compose_fig02() -> list[Path]:
    """Fig 2: All 7 panels from composite PDF (a-g)."""
    fig02_asset = REPO_ROOT / "paper/figures/fig02_svd-physical-dictionary.jpg"

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


def compose_fig03() -> list[Path]:
    """Fig 3: All 5 panels from composite PDF (a-e)."""
    fig03_asset = REPO_ROOT / "paper/figures/fig03_fingerprint-discriminability.jpg"

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


def compose_fig04() -> list[Path]:
    """Fig 4: a (external architecture JPG) on top + b,c,d (generated PDF) below."""
    fig04_asset = REPO_ROOT / "paper/figures/fig04_solver-dynamics.jpg"

    panel_a = _trim_white_border(Image.open(FIG04_PANEL_A).convert("RGB"), padding=0)
    panel_bcd = _trim_white_border(_render_pdf(FIG04_COMPOSITE_BCD), padding=4)

    target_width = 2100
    panel_a = _resize_to_width(panel_a, target_width)
    panel_bcd = _resize_to_width(panel_bcd, target_width)

    label_font = _load_font(54)
    margin = 70
    gap = 50
    canvas_w = margin * 2 + target_width
    canvas_h = margin * 2 + panel_a.height + gap + panel_bcd.height
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Panel a
    draw.text((margin, margin - 10), "a", fill="black", font=label_font)
    canvas.paste(panel_a, (margin, margin + 50))

    # Panels b,c,d (labels already in PDF)
    y_bottom = margin + panel_a.height + gap + 50
    canvas.paste(panel_bcd, (margin, y_bottom))

    _save_composite(canvas, fig04_asset)
    return [fig04_asset]


def compose_fig05() -> list[Path]:
    """Fig 5: All 6 panels from composite PDF (a-f)."""
    fig05_asset = REPO_ROOT / "paper/figures/fig05_performance-structure.jpg"

    composite = _trim_white_border(_render_pdf(FIG05_COMPOSITE, scale=4.0), padding=4)
    target_width = 2200
    composite = _resize_to_width(composite, target_width)

    margin = 40
    canvas_w = margin * 2 + composite.width
    canvas_h = margin * 2 + composite.height
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(composite, (margin, margin))

    _save_composite(canvas, fig05_asset)
    return [fig05_asset]


def compose_fig06() -> list[Path]:
    """Fig 6: a,b,c (external panel crops) on top + d,e (generated PDF) below."""
    fig06_asset = REPO_ROOT / "paper/figures/fig06_universality.jpg"

    # Load external panels a,b,c
    panel_specs = [
        ("a", "fig06_panel_a_material_exemplars.png"),
        ("b", "fig06_panel_b_heatmaps.png"),
        ("c", "fig06_panel_c_rmse_comparison.png"),
    ]

    external_panels = []
    for panel_id, filename in panel_specs:
        src = FIG06_PANELS_ABC_DIR / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing cross-material panel: {src}")
        img = Image.open(src).convert("RGB")
        img = _trim_white_border(img, padding=0)
        external_panels.append((panel_id, img))

    # Load generated d,e composite
    panel_de = _trim_white_border(_render_pdf(FIG06_COMPOSITE_DE), padding=4)

    # Layout: top section [a,b stacked left 2/3 | c right 1/3],
    #          bottom row d,e composite
    label_font = _load_font(54)
    margin = 70
    gap_x = 45
    gap_y = 50
    target_width = 2100

    # Split top into left (a+b stacked) and right (c)
    left_w = int(target_width * 0.65)
    right_w = target_width - left_w - gap_x

    _, panel_a = external_panels[0]
    _, panel_b = external_panels[1]
    _, panel_c = external_panels[2]

    panel_a = _resize_to_width(panel_a, left_w)
    panel_b = _resize_to_width(panel_b, left_w)
    left_h = panel_a.height + gap_y // 2 + panel_b.height

    panel_c = _resize_to_width(panel_c, right_w)
    # If c is taller than left stack, scale c to match
    if panel_c.height > left_h:
        scale = left_h / panel_c.height
        panel_c = panel_c.resize(
            (int(panel_c.width * scale), left_h), Image.Resampling.LANCZOS
        )

    top_h = max(left_h, panel_c.height)

    # Bottom row
    panel_de = _resize_to_width(panel_de, target_width)

    canvas_w = margin * 2 + target_width
    canvas_h = margin * 2 + top_h + gap_y + panel_de.height
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    # Top-left: a above b
    canvas.paste(panel_a, (margin, margin))
    canvas.paste(panel_b, (margin, margin + panel_a.height + gap_y // 2))

    # Top-right: c
    canvas.paste(panel_c, (margin + left_w + gap_x, margin))

    # Bottom row: d,e composite
    y_bottom = margin + top_h + gap_y
    canvas.paste(panel_de, (margin, y_bottom))

    _save_composite(canvas, fig06_asset)
    return [fig06_asset]


def main() -> None:
    created: list[Path] = []
    created.extend(compose_fig01())
    created.extend(compose_fig02())
    created.extend(compose_fig03())
    created.extend(compose_fig04())
    created.extend(compose_fig05())
    created.extend(compose_fig06())
    print("\n".join(str(path) for path in created))


if __name__ == "__main__":
    main()
