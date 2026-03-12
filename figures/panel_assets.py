"""Internal split-panel asset preparation for multi-panel figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps

from figures.style import load_paths

try:
    import fitz
except ImportError:  # pragma: no cover - optional runtime dependency
    fitz = None


@dataclass
class PanelItem:
    panel_id: str
    title: str
    provenance_mode: str
    asset_prefix: str | None
    asset_path: str | None
    bbox_norm: dict[str, float] | None


@dataclass
class FigurePanelSpec:
    figure_id: str
    composite_asset: str
    panel_manifest: str | None
    panel_dir: str | None
    storage_mode: str
    source_asset: str | None
    source_layer: str
    panels: list[PanelItem]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_specs(repo_root: Path) -> list[FigurePanelSpec]:
    cfg_path = repo_root / "figures/conf/panel_assets.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    specs: list[FigurePanelSpec] = []
    for figure_id, raw in cfg["figures"].items():
        panels = [
            PanelItem(
                panel_id=item["panel_id"],
                title=item["title"],
                provenance_mode=item["provenance_mode"],
                asset_prefix=item.get("asset_prefix"),
                asset_path=item.get("asset_path"),
                bbox_norm=item.get("bbox_norm"),
            )
            for item in raw.get("panels", [])
        ]
        specs.append(
            FigurePanelSpec(
                figure_id=figure_id,
                composite_asset=raw["composite_asset"],
                panel_manifest=raw.get("panel_manifest"),
                panel_dir=raw.get("panel_dir"),
                storage_mode=raw.get("storage_mode", "reference_existing_outputs"),
                source_asset=raw.get("source_asset"),
                source_layer=raw.get("source_layer", "manuscript_asset"),
                panels=panels,
            )
        )
    return specs


def _crop_image(source_path: Path, bbox_norm: dict[str, float], dest_path: Path) -> dict[str, Any]:
    with Image.open(source_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        width_px, height_px = img.size
        x0 = int(round(bbox_norm["x0"] * width_px))
        y0 = int(round(bbox_norm["y0"] * height_px))
        x1 = int(round((bbox_norm["x0"] + bbox_norm["width"]) * width_px))
        y1 = int(round((bbox_norm["y0"] + bbox_norm["height"]) * height_px))
        crop = img.crop((x0, y0, x1, y1))
        crop.save(dest_path)
    return {
        "width_px": x1 - x0,
        "height_px": y1 - y0,
        "bbox_px": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
    }


def _crop_pdf(source_path: Path, bbox_norm: dict[str, float], dest_path: Path) -> dict[str, Any]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required to crop PDF-backed panel assets")

    with fitz.open(str(source_path)) as doc:
        page = doc[0]
        rect = page.rect
        clip = fitz.Rect(
            rect.x0 + bbox_norm["x0"] * rect.width,
            rect.y0 + bbox_norm["y0"] * rect.height,
            rect.x0 + (bbox_norm["x0"] + bbox_norm["width"]) * rect.width,
            rect.y0 + (bbox_norm["y0"] + bbox_norm["height"]) * rect.height,
        )
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=clip, alpha=False)
        pix.save(str(dest_path))
    return {
        "width_px": pix.width,
        "height_px": pix.height,
        "bbox_pdf_points": {
            "x0": round(clip.x0, 3),
            "y0": round(clip.y0, 3),
            "x1": round(clip.x1, 3),
            "y1": round(clip.y1, 3),
        },
    }


def _write_crop_metadata(dest_path: Path, figure_id: str, panel: PanelItem, source_path: Path, crop_meta: dict[str, Any], bbox_norm: dict[str, float], source_layer: str, storage_mode: str) -> Path:
    payload = {
        "figure_id": figure_id,
        "panel_id": panel.panel_id,
        "title": panel.title,
        "source_asset": str(source_path),
        "source_layer": source_layer,
        "storage_mode": storage_mode,
        "provenance_mode": panel.provenance_mode,
        "bbox_norm": bbox_norm,
        "crop_metadata": crop_meta,
    }
    meta_path = dest_path.with_suffix(".layout.json")
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return meta_path


def _prepare_reference_manifest(repo_root: Path, spec: FigurePanelSpec) -> list[Path]:
    assert spec.panel_dir is not None
    panel_dir = repo_root / spec.panel_dir
    panel_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = panel_dir / f"{spec.figure_id}_panel_manifest.json"

    panels_payload = []
    for panel in spec.panels:
        assert panel.asset_path is not None
        panels_payload.append(
            {
                "panel_id": panel.panel_id,
                "title": panel.title,
                "provenance_mode": panel.provenance_mode,
                "storage_mode": spec.storage_mode,
                "asset_path": panel.asset_path,
                "source_asset": panel.asset_path,
                "source_layer": spec.source_layer,
            }
        )

    manifest = {
        "figure_id": spec.figure_id,
        "composite_asset": spec.composite_asset,
        "storage_mode": spec.storage_mode,
        "panel_order": [panel.panel_id for panel in spec.panels],
        "panels": panels_payload,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return [manifest_path]


def _prepare_crop_manifest(repo_root: Path, spec: FigurePanelSpec) -> list[Path]:
    assert spec.panel_dir is not None
    panel_dir = repo_root / spec.panel_dir
    panel_dir.mkdir(parents=True, exist_ok=True)

    source_rel = spec.source_asset or spec.composite_asset
    source_path = repo_root / source_rel
    if not source_path.exists():
        print(f"[panel-assets] SKIP {spec.figure_id}: source asset not found at {source_path}")
        return []

    created: list[Path] = []
    panels_payload = []
    for panel in spec.panels:
        assert panel.asset_prefix is not None
        assert panel.bbox_norm is not None
        out_path = panel_dir / f"{panel.asset_prefix}.png"
        if source_path.suffix.lower() == ".pdf":
            crop_meta = _crop_pdf(source_path, panel.bbox_norm, out_path)
        else:
            crop_meta = _crop_image(source_path, panel.bbox_norm, out_path)
        meta_path = _write_crop_metadata(
            out_path,
            spec.figure_id,
            panel,
            source_path,
            crop_meta,
            panel.bbox_norm,
            spec.source_layer,
            spec.storage_mode,
        )
        created.extend([out_path, meta_path])
        panels_payload.append(
            {
                "panel_id": panel.panel_id,
                "title": panel.title,
                "provenance_mode": panel.provenance_mode,
                "storage_mode": spec.storage_mode,
                "asset_path": str(out_path.relative_to(repo_root)),
                "source_asset": source_rel,
                "source_layer": spec.source_layer,
                "bbox_norm": panel.bbox_norm,
                "metadata_path": str(meta_path.relative_to(repo_root)),
            }
        )

    manifest_path = panel_dir / f"{spec.figure_id}_panel_manifest.json"
    manifest = {
        "figure_id": spec.figure_id,
        "composite_asset": spec.composite_asset,
        "storage_mode": spec.storage_mode,
        "panel_order": [panel.panel_id for panel in spec.panels],
        "panels": panels_payload,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    created.append(manifest_path)
    return created


def prepare_all(repo_root: Path | None = None) -> list[Path]:
    repo_root = _repo_root() if repo_root is None else repo_root
    _ = load_paths()
    created: list[Path] = []
    for spec in _load_specs(repo_root):
        if spec.panel_manifest is not None:
            manifest_path = repo_root / spec.panel_manifest
            if manifest_path.exists():
                created.append(manifest_path)
                continue
            print(f"[panel-assets] WARN {spec.figure_id}: expected manifest missing at {manifest_path}")
            continue
        if spec.storage_mode == "reference_existing_outputs":
            created.extend(_prepare_reference_manifest(repo_root, spec))
            continue
        created.extend(_prepare_crop_manifest(repo_root, spec))

    print(f"[panel-assets] Prepared {len(created)} panel asset file(s)")
    return created


def main() -> None:
    prepare_all()


if __name__ == "__main__":
    main()
