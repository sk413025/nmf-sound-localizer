#!/usr/bin/env python3
"""Compose panel-first paper-facing figures.

This script rebuilds the manuscript assets for the current main-paper figure set
from panel-level assets whenever possible:

- Fig. 1: setup + spectral fingerprint
- Fig. 2: SVD / modal decomposition / dictionary
- Fig. 4: ablation + SNR robustness
- Fig. 5: structure + macro selection robustness
- Fig. 6: angle-specific routing diagnostics
- Fig. 7: band-wise routing diagnostics I
- Fig. 8: band-wise routing diagnostics II
- Fig. 9: cross-material universality

It also syncs lineage-bundle preview copies for the expanded Master Figure 3
family under ``docs/working-notes/`` and ``results/``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

try:
    import fitz
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("PyMuPDF is required for figure composition") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_BUNDLE_ROOT = REPO_ROOT / "docs/working-notes/master_figure3_lineage_bundle_20260312"
LOCAL_BUNDLE_ROOT = REPO_ROOT / "results/master_figure3_lineage_bundle_20260312"

FIG01_PANEL_A = REPO_ROOT / "figures/output/fig01_paradigm_shift_panels/fig01_panel_a_experimental_setup.png"
FIG01_PANEL_B = REPO_ROOT / "figures/output/fig01_paradigm_shift_panels/fig01_panel_b_spectral_fingerprint.png"
FIG02_PANEL_A = REPO_ROOT / "figures/output/fig02_svd_spectrum_panels/fig02_panel_a_singular_values.pdf"
FIG02_PANEL_B = REPO_ROOT / "figures/output/fig02_svd_spectrum_panels/fig02_panel_b_modal_decomposition.pdf"
FIG02_PANEL_C = REPO_ROOT / "figures/output/fig02_svd_spectrum_panels/fig02_panel_c_dictionary_modes.pdf"
FIG04_PANEL_A = REPO_ROOT / "figures/output/fig04_snr_ablation_panels/fig04_panel_a_ablation_strip.pdf"
FIG04_PANEL_B = REPO_ROOT / "figures/output/fig04_snr_ablation_panels/fig04_panel_b_snr_curves.pdf"
FIG05_SOURCE = REPO_ROOT / "figures/output/fig05_structure_alignment.tiff"
FIG05_PANEL_A = REPO_ROOT / "figures/output/fig05_routing_mechanism_analysis_panels/fig05_panel_a_global_attention.pdf"
FIG05_PANEL_B = REPO_ROOT / "figures/output/fig05_routing_mechanism_analysis_panels/fig05_panel_c_macro_robustness.pdf"
FIG06_ANGLE55 = REPO_ROOT / "figures/output/fig09_angle55.pdf"
FIG06_ANGLE100 = REPO_ROOT / "figures/output/fig09_angle100.pdf"
FIG09_LEGACY_PANEL_DIR = REPO_ROOT / "figures/output/fig06_cross_material_universality_panels"

FIG07_PARTS = [
    (
        "a",
        "Full-band (smoothed)",
        REPO_ROOT / "results/fig5_b3_line5_20260202_224349/Fig5_B3_LINE_300_3000.pdf",
        "fig07_panel_a_fullband_smooth.pdf",
    ),
    (
        "b",
        "Full-band (no smoothing)",
        REPO_ROOT / "results/fig5_b3_line5_nosmooth_20260202_233434/Fig5_B3_LINE_300_3000.pdf",
        "fig07_panel_b_fullband_nosmooth.pdf",
    ),
    (
        "c",
        "300-500 Hz",
        REPO_ROOT / "results/fig5_b3_line5_nosmooth_20260202_233434/Fig5_B3_LINE_300_500.pdf",
        "fig07_panel_c_300_500.pdf",
    ),
]

FIG08_PARTS = [
    (
        "a",
        "500-1000 Hz",
        REPO_ROOT / "results/fig5_b3_line5_nosmooth_20260202_233434/Fig5_B3_LINE_500_1000.pdf",
        "fig08_panel_a_500_1000.pdf",
    ),
    (
        "b",
        "1000-2000 Hz",
        REPO_ROOT / "results/fig5_b3_line5_nosmooth_20260202_233434/Fig5_B3_LINE_1000_2000.pdf",
        "fig08_panel_b_1000_2000.pdf",
    ),
    (
        "c",
        "2000-3000 Hz",
        REPO_ROOT / "results/fig5_b3_line5_nosmooth_20260202_233434/Fig5_B3_LINE_2000_3000.pdf",
        "fig08_panel_c_2000_3000.pdf",
    ),
]


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


def _crop_pixels(img: Image.Image, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> Image.Image:
    return img.crop((left, top, max(img.width - right, left + 1), max(img.height - bottom, top + 1)))


def _cover_with_white(img: Image.Image, left: int, top: int, right: int, bottom: int) -> Image.Image:
    patched = img.copy()
    draw = ImageDraw.Draw(patched)
    draw.rectangle((left, top, right, bottom), fill="white")
    return patched


def _render_pdf(pdf_path: Path, scale: float = 3.0) -> Image.Image:
    with fitz.open(str(pdf_path)) as doc:
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _resize_to_width(img: Image.Image, target_width: int) -> Image.Image:
    scale = target_width / max(img.width, 1)
    target_height = int(round(img.height * scale))
    return img.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _save_composite(img: Image.Image, path: Path) -> None:
    _ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        img.save(path, format="JPEG", quality=95, subsampling=0)
    elif suffix == ".png":
        img.save(path, format="PNG")
    else:
        raise ValueError(f"Unsupported composite suffix for {path}")


def _write_manifest(manifest_path: Path, payload: dict) -> None:
    _ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def compose_fig01() -> list[Path]:
    fig01_asset = REPO_ROOT / "paper/figures/fig01_paradigm-shift.jpg"
    left_panel = _trim_white_border(Image.open(FIG01_PANEL_A).convert("RGB"), padding=0)
    right_panel = _trim_white_border(Image.open(FIG01_PANEL_B).convert("RGB"), padding=0)

    target_width = 970
    left_panel = _resize_to_width(left_panel, target_width)
    right_panel = _resize_to_width(right_panel, target_width)

    margin = 70
    gap = 70
    canvas_w = margin * 2 + target_width * 2 + gap
    canvas_h = margin * 2 + max(left_panel.height, right_panel.height)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(left_panel, (margin, margin))
    canvas.paste(right_panel, (margin + target_width + gap, margin))
    _save_composite(canvas, fig01_asset)
    return [fig01_asset]


def compose_fig02() -> list[Path]:
    fig02_asset = REPO_ROOT / "paper/figures/fig02_svd-physical-dictionary.jpg"
    panel_a = _trim_white_border(_render_pdf(FIG02_PANEL_A), padding=4)
    panel_b = _trim_white_border(_render_pdf(FIG02_PANEL_B), padding=4)
    panel_c = _trim_white_border(_render_pdf(FIG02_PANEL_C), padding=4)
    panel_a = _crop_pixels(panel_a, left=18, top=18)
    panel_b = _crop_pixels(panel_b, left=18, top=18)
    panel_c = _crop_pixels(panel_c, left=18, top=18)

    widths = [600, 760, 760]
    panel_a = _resize_to_width(panel_a, widths[0])
    panel_b = _resize_to_width(panel_b, widths[1])
    panel_c = _resize_to_width(panel_c, widths[2])

    label_font = _load_font(54)
    margin = 65
    header_h = 70
    gap = 45
    canvas_w = margin * 2 + sum(widths) + gap * 2
    canvas_h = margin * 2 + header_h + max(panel_a.height, panel_b.height, panel_c.height)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    x_positions = [
        margin,
        margin + widths[0] + gap,
        margin + widths[0] + widths[1] + gap * 2,
    ]
    for label, x in zip(["a", "b", "c"], x_positions):
        draw.text((x, margin), label, fill="black", font=label_font)
    y = margin + header_h
    canvas.paste(panel_a, (x_positions[0], y))
    canvas.paste(panel_b, (x_positions[1], y))
    canvas.paste(panel_c, (x_positions[2], y))
    _save_composite(canvas, fig02_asset)
    return [fig02_asset]


def compose_fig04() -> list[Path]:
    fig04_asset = REPO_ROOT / "paper/figures/fig04_noise-robustness-ablation.jpg"
    panel_a = _trim_white_border(_render_pdf(FIG04_PANEL_A), padding=4)
    panel_b = _trim_white_border(_render_pdf(FIG04_PANEL_B), padding=4)
    panel_a = _crop_pixels(panel_a, left=18, top=18)
    panel_b = _crop_pixels(panel_b, left=18, top=18)

    widths = [900, 1020]
    panel_a = _resize_to_width(panel_a, widths[0])
    panel_b = _resize_to_width(panel_b, widths[1])

    label_font = _load_font(54)
    margin = 70
    header_h = 70
    gap = 70
    canvas_w = margin * 2 + sum(widths) + gap
    canvas_h = margin * 2 + header_h + max(panel_a.height, panel_b.height)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    x_positions = [margin, margin + widths[0] + gap]
    draw.text((x_positions[0], margin), "a", fill="black", font=label_font)
    draw.text((x_positions[1], margin), "b", fill="black", font=label_font)
    y = margin + header_h
    canvas.paste(panel_a, (x_positions[0], y))
    canvas.paste(panel_b, (x_positions[1], y))
    _save_composite(canvas, fig04_asset)
    return [fig04_asset]


def compose_fig05() -> list[Path]:
    fig05_asset = REPO_ROOT / "paper/figures/fig05_structure-macro-selection.png"
    panel_dir = REPO_ROOT / "figures/output/fig05_structure_macro_panels"
    bundle_dir = DOCS_BUNDLE_ROOT / "01_current_canonical"
    local_bundle_dir = LOCAL_BUNDLE_ROOT / "01_current_canonical"
    panel_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    local_bundle_dir.mkdir(parents=True, exist_ok=True)
    left_panel = _trim_white_border(_render_pdf(FIG05_PANEL_A), padding=6)
    right_panel = _trim_white_border(_render_pdf(FIG05_PANEL_B), padding=6)
    left_panel = left_panel.crop((70, 0, left_panel.width, left_panel.height))
    right_panel = right_panel.crop((70, 0, right_panel.width, right_panel.height))
    target_width = 1020
    left_panel = _resize_to_width(left_panel, target_width)
    right_panel = _resize_to_width(right_panel, target_width)

    label_font = _load_font(54)
    margin = 70
    header_h = 70
    gap = 70
    canvas_w = margin * 2 + target_width * 2 + gap
    canvas_h = margin * 2 + header_h + max(left_panel.height, right_panel.height)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, margin), "a", fill="black", font=label_font)
    draw.text((margin + target_width + gap, margin), "b", fill="black", font=label_font)
    canvas.paste(left_panel, (margin, margin + header_h))
    canvas.paste(right_panel, (margin + target_width + gap, margin + header_h))

    _save_composite(canvas, fig05_asset)
    shutil.copy2(fig05_asset, bundle_dir / "current_fig05_composite.png")
    shutil.copy2(fig05_asset, bundle_dir / "current_fig05_structure_macro.png")
    shutil.copy2(fig05_asset, local_bundle_dir / "current_fig05_composite.png")
    shutil.copy2(fig05_asset, local_bundle_dir / "current_fig05_structure_macro.png")

    manifest = {
        "figure_id": "fig05",
        "composite_asset": "paper/figures/fig05_structure-macro-selection.png",
        "storage_mode": "reference_existing_outputs",
        "panel_order": ["a", "b"],
        "panels": [
            {
                "panel_id": "a",
                "title": "Global structure alignment",
                "provenance_mode": "data_backed",
                "storage_mode": "reference_existing_outputs",
                "asset_path": "figures/output/fig05_routing_mechanism_analysis_panels/fig05_panel_a_global_attention.pdf",
                "source_asset": "figures/output/fig05_routing_mechanism_analysis_panels/fig05_panel_a_global_attention.pdf",
                "source_layer": "generator_output",
            },
            {
                "panel_id": "b",
                "title": "Macro selection robustness",
                "provenance_mode": "data_backed",
                "storage_mode": "reference_existing_outputs",
                "asset_path": "figures/output/fig05_routing_mechanism_analysis_panels/fig05_panel_c_macro_robustness.pdf",
                "source_asset": "figures/output/fig05_routing_mechanism_analysis_panels/fig05_panel_c_macro_robustness.pdf",
                "source_layer": "generator_output",
            },
        ],
    }
    manifest_path = panel_dir / "fig05_panel_manifest.json"
    _write_manifest(manifest_path, manifest)
    return [fig05_asset, manifest_path]


def compose_fig06() -> list[Path]:
    fig06_asset = REPO_ROOT / "paper/figures/fig06_angle-specific-mechanism.png"
    panel_dir = REPO_ROOT / "figures/output/fig06_angle_specific_mechanism_panels"
    bundle_dir = DOCS_BUNDLE_ROOT / "01_current_canonical"
    local_bundle_dir = LOCAL_BUNDLE_ROOT / "01_current_canonical"
    panel_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    local_bundle_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("a", "55-degree routing distribution", _trim_white_border(_render_pdf(FIG06_ANGLE55), padding=6)),
        ("b", "100-degree routing distribution", _trim_white_border(_render_pdf(FIG06_ANGLE100), padding=6)),
    ]

    label_font = _load_font(54)
    target_width = 2100
    resized = [(panel_id, title, _resize_to_width(img, target_width)) for panel_id, title, img in panels]
    margin = 70
    header_h = 70
    gap = 70
    canvas_h = margin * 2 + header_h * len(resized) + sum(img.height for _, _, img in resized) + gap
    canvas = Image.new("RGB", (target_width + margin * 2, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    y = margin + header_h
    for index, (panel_id, _, img) in enumerate(resized):
        draw.text((margin, y - 56), panel_id, fill="black", font=label_font)
        canvas.paste(img, (margin, y))
        y += img.height + gap + (header_h if index == 0 else 0)

    _save_composite(canvas, fig06_asset)
    shutil.copy2(fig06_asset, bundle_dir / "current_fig06_angle_specific.png")
    shutil.copy2(fig06_asset, local_bundle_dir / "current_fig06_angle_specific.png")

    manifest = {
        "figure_id": "fig06",
        "composite_asset": "paper/figures/fig06_angle-specific-mechanism.png",
        "storage_mode": "reference_existing_outputs",
        "panel_order": ["a", "b"],
        "panels": [
            {
                "panel_id": "a",
                "title": "Angle-specific routing distribution at 55 degrees",
                "provenance_mode": "data_backed",
                "storage_mode": "reference_existing_outputs",
                "asset_path": "figures/output/fig09_angle55.pdf",
                "source_asset": "figures/output/fig09_angle55.pdf",
                "source_layer": "generator_output",
            },
            {
                "panel_id": "b",
                "title": "Angle-specific routing distribution at 100 degrees",
                "provenance_mode": "data_backed",
                "storage_mode": "reference_existing_outputs",
                "asset_path": "figures/output/fig09_angle100.pdf",
                "source_asset": "figures/output/fig09_angle100.pdf",
                "source_layer": "generator_output",
            },
        ],
    }
    manifest_path = panel_dir / "fig06_panel_manifest.json"
    _write_manifest(manifest_path, manifest)
    return [fig06_asset, manifest_path]


def _compose_bandwise(
    figure_id: str,
    output_name: str,
    bundle_name: str,
    panel_dir_name: str,
    specs: list[tuple[str, str, Path, str]],
) -> list[Path]:
    output_path = REPO_ROOT / "paper/figures" / output_name
    panel_dir = REPO_ROOT / "figures/output" / panel_dir_name
    bundle_dir = DOCS_BUNDLE_ROOT / "01_current_canonical"
    local_bundle_dir = LOCAL_BUNDLE_ROOT / "01_current_canonical"
    panel_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    local_bundle_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[tuple[str, str, Image.Image, Path]] = []
    created: list[Path] = []
    for panel_id, title, source_pdf, copied_name in specs:
        if not source_pdf.exists():
            raise FileNotFoundError(f"Missing source PDF for {figure_id}: {source_pdf}")
        copied_pdf = panel_dir / copied_name
        shutil.copy2(source_pdf, copied_pdf)
        img = _trim_white_border(_render_pdf(source_pdf), padding=6)
        img = _crop_pixels(img, top=95)
        rendered.append((panel_id, title, img, copied_pdf))
        created.append(copied_pdf)

    target_width = 740
    margin = 70
    gap = 45
    header_h = 105
    label_font = _load_font(54)
    title_font = _load_font(22)
    resized = [(panel_id, title, _resize_to_width(img, target_width), copied_pdf) for panel_id, title, img, copied_pdf in rendered]
    max_h = max(img.height for _, _, img, _ in resized)
    canvas_w = margin * 2 + target_width * 3 + gap * 2
    canvas_h = margin * 2 + header_h + max_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (panel_id, title, img, _) in enumerate(resized):
        x = margin + idx * (target_width + gap)
        y = margin + header_h
        draw.text((x, y - 74), panel_id, fill="black", font=label_font)
        draw.text((x + 34, y - 68), title, fill="black", font=title_font)
        canvas.paste(img, (x, y))

    _save_composite(canvas, output_path)
    shutil.copy2(output_path, bundle_dir / bundle_name)
    shutil.copy2(output_path, local_bundle_dir / bundle_name)
    created.append(output_path)

    manifest = {
        "figure_id": figure_id,
        "composite_asset": str(output_path.relative_to(REPO_ROOT)),
        "storage_mode": "reference_existing_outputs",
        "panel_order": [panel_id for panel_id, _, _, _ in specs],
        "panels": [
            {
                "panel_id": panel_id,
                "title": title,
                "provenance_mode": "data_backed",
                "storage_mode": "reference_existing_outputs",
                "asset_path": str((panel_dir / copied_name).relative_to(REPO_ROOT)),
                "source_asset": str(source_pdf.relative_to(REPO_ROOT)),
                "source_layer": "results_pdf",
            }
            for panel_id, title, source_pdf, copied_name in specs
        ],
    }
    manifest_path = panel_dir / f"{figure_id}_panel_manifest.json"
    _write_manifest(manifest_path, manifest)
    created.append(manifest_path)
    return created


def compose_fig07() -> list[Path]:
    return _compose_bandwise(
        figure_id="fig07",
        output_name="fig07_bandwise-routing-analysis-part1.png",
        bundle_name="current_fig07_bandwise_part1.png",
        panel_dir_name="fig07_bandwise_routing_part1_panels",
        specs=FIG07_PARTS,
    )


def compose_fig08() -> list[Path]:
    return _compose_bandwise(
        figure_id="fig08",
        output_name="fig08_bandwise-routing-analysis-part2.png",
        bundle_name="current_fig08_bandwise_part2.png",
        panel_dir_name="fig08_bandwise_routing_part2_panels",
        specs=FIG08_PARTS,
    )


def compose_fig09() -> list[Path]:
    fig09_asset = REPO_ROOT / "paper/figures/fig09_cross-material-universality.jpg"
    panel_dir = REPO_ROOT / "figures/output/fig09_cross_material_universality_panels"
    bundle_dir = DOCS_BUNDLE_ROOT / "04_separate_families"
    local_bundle_dir = LOCAL_BUNDLE_ROOT / "04_separate_families"
    panel_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    local_bundle_dir.mkdir(parents=True, exist_ok=True)
    panel_specs = [
        ("a", "Material exemplars", "fig06_panel_a_material_exemplars.png", "fig09_panel_a_material_exemplars.png"),
        ("b", "Cross-material heatmaps", "fig06_panel_b_heatmaps.png", "fig09_panel_b_heatmaps.png"),
        ("c", "RMSE comparison", "fig06_panel_c_rmse_comparison.png", "fig09_panel_c_rmse_comparison.png"),
    ]
    panels = []
    created: list[Path] = []
    rendered: list[tuple[str, str, Image.Image, Path]] = []
    for panel_id, title, src_name, dst_name in panel_specs:
        src_path = FIG09_LEGACY_PANEL_DIR / src_name
        dst_path = panel_dir / dst_name
        if not src_path.exists():
            raise FileNotFoundError(f"Missing cross-material panel asset: {src_path}")
        shutil.copy2(src_path, dst_path)
        img = Image.open(dst_path).convert("RGB")
        if panel_id == "a":
            img = _crop_pixels(img, left=14, top=66, bottom=26)
        elif panel_id == "b":
            img = _crop_pixels(img, left=14, top=58, bottom=68)
            img = _cover_with_white(img, 0, 0, 64, 52)
        else:
            img = _crop_pixels(img, left=14, top=42, bottom=14)
        img = _trim_white_border(img, padding=0)
        rendered.append((panel_id, title, img, dst_path))
        panels.append(
            {
                "panel_id": panel_id,
                "title": title,
                "provenance_mode": "provenance_gap",
                "storage_mode": "reference_existing_outputs",
                "asset_path": str(dst_path.relative_to(REPO_ROOT)),
                "source_asset": str((FIG09_LEGACY_PANEL_DIR / src_name).relative_to(REPO_ROOT)),
                "source_layer": "legacy_split_panel_asset",
            }
        )
        created.append(dst_path)

    panel_map = {panel_id: (title, img, path) for panel_id, title, img, path in rendered}
    panel_a = _resize_to_width(panel_map["a"][1], 1730)
    panel_b = _resize_to_width(panel_map["b"][1], 1730)
    panel_c = _resize_to_width(panel_map["c"][1], 620)

    label_font = _load_font(54)
    margin = 70
    header_h = 70
    gap_x = 55
    gap_y = 48
    left_w = max(panel_a.width, panel_b.width)
    canvas_w = margin * 2 + left_w + gap_x + panel_c.width
    canvas_h = margin * 2 + header_h * 2 + panel_a.height + gap_y + max(panel_b.height, panel_c.height)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    ax_x = margin
    bx_x = margin
    cx_x = margin + left_w + gap_x
    ay = margin + header_h
    by = ay + panel_a.height + gap_y + header_h
    cy = margin + header_h
    draw.text((ax_x, margin), "a", fill="black", font=label_font)
    draw.text((bx_x, ay + panel_a.height + gap_y), "b", fill="black", font=label_font)
    draw.text((cx_x, margin), "c", fill="black", font=label_font)
    canvas.paste(panel_a, (ax_x, ay))
    canvas.paste(panel_b, (bx_x, by))
    canvas.paste(panel_c, (cx_x, cy))

    _save_composite(canvas, fig09_asset)
    shutil.copy2(fig09_asset, bundle_dir / "current_fig09_cross_material.jpg")
    shutil.copy2(fig09_asset, local_bundle_dir / "current_fig09_cross_material.jpg")
    created.insert(0, fig09_asset)

    manifest = {
        "figure_id": "fig09",
        "composite_asset": "paper/figures/fig09_cross-material-universality.jpg",
        "storage_mode": "reference_existing_outputs",
        "panel_order": ["a", "b", "c"],
        "panels": panels,
    }
    manifest_path = panel_dir / "fig09_panel_manifest.json"
    _write_manifest(manifest_path, manifest)
    created.append(manifest_path)
    return created


def main() -> None:
    created: list[Path] = []
    created.extend(compose_fig01())
    created.extend(compose_fig02())
    created.extend(compose_fig04())
    created.extend(compose_fig05())
    created.extend(compose_fig06())
    created.extend(compose_fig07())
    created.extend(compose_fig08())
    created.extend(compose_fig09())
    print("\n".join(str(path) for path in created))


if __name__ == "__main__":
    main()
