"""Codex-governed review bundle generation and gating for paper-facing figures."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageOps

from figures.style import DOUBLE_COL_MM, SINGLE_COL_MM

try:
    import fitz
except ImportError:  # pragma: no cover - optional runtime dependency
    fitz = None


ROLE_EXPECTATIONS = {
    "main": {"main"},
    "supp": {"supp"},
    "extended": {"supp", "extended"},
}
REQUIRED_REVIEW_ROLES = ("visual-reviewer", "manuscript-fit-reviewer", "supervisor")


@dataclass
class ReviewTarget:
    figure_id: str
    role: str
    asset: str
    asset_layer: str
    provenance_mode: str
    width_mode: str
    manuscript_section: str
    manuscript_path: str
    manuscript_anchor: str
    legend_path: str
    registry_path: str
    claim: str
    panel_order: list[str]
    panel_manifest: str | None
    legend_summary: str
    evidence_sources: list[str]
    generator_outputs: list[str]
    provenance_note: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _width_mm_for_mode(width_mode: str) -> float:
    return float(SINGLE_COL_MM if width_mode == "single" else DOUBLE_COL_MM)


def _load_targets(repo_root: Path) -> tuple[Path, Path, list[ReviewTarget]]:
    cfg_path = repo_root / "figures/conf/review_targets.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    review_dir = repo_root / cfg.get("review_dir", "figures/review_artifacts")
    canonical = repo_root / cfg["canonical_requirements"]
    targets = [
        ReviewTarget(
            figure_id=item["figure_id"],
            role=item["role"],
            asset=item["asset"],
            asset_layer=item.get("asset_layer", "manuscript_composite"),
            provenance_mode=item.get("provenance_mode", "unspecified"),
            width_mode=item.get("width_mode", "double"),
            manuscript_section=item.get("manuscript_section", "Results"),
            manuscript_path=item.get("manuscript_path", "paper/manuscript/manuscript.md"),
            manuscript_anchor=item.get("manuscript_anchor", item.get("manuscript_section", "Results")),
            legend_path=item.get("legend_path", "paper/figures/Figure-Legends.md"),
            registry_path=item.get("registry_path", "figures/FIGURE_REGISTRY.md"),
            claim=item["claim"],
            panel_order=list(item.get("panel_order", [])),
            panel_manifest=item.get("panel_manifest"),
            legend_summary=item.get("legend_summary", ""),
            evidence_sources=list(item.get("evidence_sources", [])),
            generator_outputs=list(item.get("generator_outputs", [])),
            provenance_note=item.get("provenance_note", ""),
        )
        for item in cfg["targets"]
    ]
    return review_dir, canonical, targets


def _load_layout_metadata(repo_root: Path, stem: str) -> dict[str, Any] | None:
    for candidate in (
        repo_root / "paper/figures" / f"{stem}.layout.json",
        repo_root / "figures/output" / f"{stem}.layout.json",
    ):
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                return json.load(f)
    return None


def _load_panel_manifest(repo_root: Path, panel_manifest: str | None) -> dict[str, Any] | None:
    if panel_manifest is None:
        return None
    manifest_path = repo_root / panel_manifest
    if not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _render_pdf_preview(pdf_path: Path, out_path: Path, max_width_px: int = 1800) -> None:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required to render PDF previews")

    with fitz.open(str(pdf_path)) as doc:
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    if img.width > max_width_px:
        scale = max_width_px / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    img.save(out_path)


def _render_pdf_page_previews(pdf_path: Path, pages_dir: Path, max_width_px: int = 1800) -> list[str]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required to render PDF previews")

    pages_dir.mkdir(parents=True, exist_ok=True)
    rel_paths: list[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_index, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            if img.width > max_width_px:
                scale = max_width_px / img.width
                img = img.resize((int(img.width * scale), int(img.height * scale)))
            out_path = pages_dir / f"page_{page_index + 1:03d}.png"
            img.save(out_path)
            rel_paths.append(out_path.name)
    return rel_paths


def _render_image_preview(image_path: Path, out_path: Path, max_width_px: int = 1800) -> None:
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        if img.width > max_width_px:
            scale = max_width_px / img.width
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        img.save(out_path)


def _geometry_for_asset(asset_path: Path, width_mm: float) -> dict[str, Any]:
    if asset_path.suffix.lower() == ".pdf":
        if fitz is None:
            raise RuntimeError("PyMuPDF is required to measure PDF geometry")
        with fitz.open(str(asset_path)) as doc:
            rect = doc[0].mediabox
        return {
            "source": "pdf",
            "width_mm": round(rect.width * 25.4 / 72.0, 3),
            "height_mm": round(rect.height * 25.4 / 72.0, 3),
        }

    with Image.open(asset_path) as img:
        img = ImageOps.exif_transpose(img)
        width_px, height_px = img.size
    height_mm = width_mm * (height_px / max(width_px, 1))
    return {
        "source": "intended_width_plus_aspect",
        "width_mm": round(width_mm, 3),
        "height_mm": round(height_mm, 3),
        "width_px": width_px,
        "height_px": height_px,
    }


def _geometry_findings(target: ReviewTarget, geometry: dict[str, Any], layout_meta: dict[str, Any] | None) -> list[str]:
    findings: list[str] = []
    width_mm = geometry["width_mm"]
    height_mm = geometry["height_mm"]
    if abs(width_mm - SINGLE_COL_MM) > 1.0 and abs(width_mm - DOUBLE_COL_MM) > 1.0:
        findings.append(f"Figure width {width_mm:.1f} mm is not close to 89 or 183 mm.")
    if target.role == "main" and height_mm > 170.0:
        findings.append(f"Main figure height {height_mm:.1f} mm exceeds the 170 mm project limit.")
    if layout_meta is not None:
        for ax in layout_meta.get("axes", []):
            bbox = ax.get("bbox_mm", {})
            if bbox.get("width", 999.0) < 25.0 or bbox.get("height", 999.0) < 22.0:
                findings.append(
                    f"Axis {ax.get('index')} is very small ({bbox.get('width', 0):.1f} x {bbox.get('height', 0):.1f} mm)."
                )
    return findings


def _draw_overlay(preview_path: Path, overlay_path: Path, geometry: dict[str, Any], layout_meta: dict[str, Any] | None) -> None:
    with Image.open(preview_path) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    width_px, height_px = canvas.size
    width_mm = max(geometry["width_mm"], 1.0)
    height_mm = max(geometry["height_mm"], 1.0)

    for step_mm in range(10, int(width_mm), 10):
        x = int(width_px * (step_mm / width_mm))
        draw.line([(x, 0), (x, height_px)], fill=(120, 120, 120, 90), width=1)
    for step_mm in range(10, int(height_mm), 10):
        y = int(height_px * (step_mm / height_mm))
        draw.line([(0, y), (width_px, y)], fill=(120, 120, 120, 90), width=1)

    draw.rectangle([(1, 1), (width_px - 2, height_px - 2)], outline=(30, 120, 255, 220), width=3)
    if layout_meta is not None:
        for ax in layout_meta.get("axes", []):
            bbox = ax.get("bbox_mm", {})
            x0 = width_px * (bbox.get("x0", 0.0) / width_mm)
            y0_bottom = height_px * (bbox.get("y0", 0.0) / height_mm)
            w = width_px * (bbox.get("width", 0.0) / width_mm)
            h = height_px * (bbox.get("height", 0.0) / height_mm)
            top = height_px - (y0_bottom + h)
            box = [(x0, top), (x0 + w, top + h)]
            outline = (255, 90, 40, 210) if ax.get("has_data") else (40, 170, 255, 180)
            draw.rectangle(box, outline=outline, width=3)
            draw.text((x0 + 4, max(top + 4, 0)), f"ax{ax.get('index')}", fill=(20, 20, 20, 255))
    canvas.save(overlay_path)


def _compute_bundle_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _prepare_panel_previews(bundle_dir: Path, repo_root: Path, panel_manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if panel_manifest is None:
        return []
    previews_dir = bundle_dir / "panel_previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, Any]] = []
    for panel in panel_manifest.get("panels", []):
        asset_rel = panel.get("asset_path")
        if not asset_rel:
            continue
        asset_path = repo_root / asset_rel
        if not asset_path.exists():
            continue
        preview_name = f"{panel.get('panel_id', 'panel')}.png"
        preview_path = previews_dir / preview_name
        if asset_path.suffix.lower() == ".pdf":
            _render_pdf_preview(asset_path, preview_path, max_width_px=1200)
            pages_dir = previews_dir / f"{panel.get('panel_id', 'panel')}_pages"
            page_names = _render_pdf_page_previews(asset_path, pages_dir, max_width_px=1200)
            page_preview_paths = [str((pages_dir / name).relative_to(bundle_dir)) for name in page_names]
        else:
            _render_image_preview(asset_path, preview_path, max_width_px=1200)
            page_preview_paths = [str(preview_path.relative_to(bundle_dir))]
        prepared.append(
            {
                "panel_id": panel.get("panel_id"),
                "title": panel.get("title"),
                "asset_path": asset_rel,
                "preview_path": str(preview_path.relative_to(bundle_dir)),
                "page_preview_paths": page_preview_paths,
                "storage_mode": panel.get("storage_mode"),
                "provenance_mode": panel.get("provenance_mode"),
            }
        )
    return prepared


def _write_role_templates(bundle_dir: Path, bundle_hash: str, canonical_rel: str) -> None:
    reviews_dir = bundle_dir / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "skill_used": "paper-asset-review",
        "canonical_requirements_path": canonical_rel,
        "bundle_hash": bundle_hash,
        "review_timestamp": "",
        "reviewer_identity": "",
    }
    templates = {
        "visual-reviewer": {
            **base,
            "reviewer_role": "visual-reviewer",
            "overall_verdict": "pass|fail",
            "paper_role_fit": "main|supp|extended|neither",
            "inspection_complete": False,
            "visual_assets_inspected": [],
            "pdf_pages_converted_and_inspected": [],
            "generator_paths_inspected": [],
            "evidence_paths_inspected": [],
            "typography_issues": [],
            "semantic_color_issues": [],
            "narration_issues": [],
            "readability_issues": [],
            "hierarchy_issues": [],
            "style_consistency_notes": [],
            "white_space_efficiency_notes": [],
            "recommended_width_mode": "single|double",
            "recommended_height_mm": 0,
            "panel_changes": [],
            "primary_reason": "",
        },
        "manuscript-fit-reviewer": {
            **base,
            "reviewer_role": "manuscript-fit-reviewer",
            "overall_verdict": "pass|fail",
            "paper_role_fit": "main|supp|extended|neither",
            "inspection_complete": False,
            "visual_assets_inspected": [],
            "pdf_pages_converted_and_inspected": [],
            "generator_paths_inspected": [],
            "evidence_paths_inspected": [],
            "claim_support_assessment": "",
            "claim_mismatch_issues": [],
            "caption_delegation_issues": [],
            "move_to_supplementary": False,
            "split_recommended": False,
            "panel_changes": [],
            "primary_reason": "",
        },
        "supervisor": {
            **base,
            "reviewer_role": "supervisor",
            "overall_verdict": "pass|fail",
            "paper_role_fit": "main|supp|extended|neither",
            "consolidated_from_roles": ["visual-reviewer", "manuscript-fit-reviewer"],
            "inspection_complete": False,
            "visual_assets_inspected": [],
            "pdf_pages_converted_and_inspected": [],
            "generator_paths_inspected": [],
            "evidence_paths_inspected": [],
            "primary_reason": "",
            "typography_issues": [],
            "semantic_color_issues": [],
            "narration_issues": [],
            "readability_issues": [],
            "hierarchy_issues": [],
            "claim_mismatch_issues": [],
            "caption_delegation_issues": [],
            "recommended_width_mode": "single|double",
            "recommended_height_mm": 0,
            "panel_changes": [],
            "split_recommended": False,
            "move_to_supplementary": False,
            "style_consistency_notes": [],
            "white_space_efficiency_notes": [],
        },
    }
    for role, payload in templates.items():
        (reviews_dir / f"{role}.template.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_workflow(bundle_dir: Path, target: ReviewTarget, canonical_rel: str, bundle_hash: str) -> None:
    workflow = {
        "entrypoint_script": "scripts/paper/review_paper_assets.py",
        "required_skill": "paper-asset-review",
        "bundle_hash": bundle_hash,
        "canonical_requirements_path": canonical_rel,
        "required_roles": list(REQUIRED_REVIEW_ROLES),
        "required_inspection_order": [
            "inspect_manuscript_asset_visually",
            "convert_and_inspect_all_pdf_pages",
            "inspect_split_panel_or_upstream_figure_assets",
            "inspect_generator_or_composition_code",
            "inspect_evidence_or_provenance_sources",
            "then_judge_manuscript_fit_and_release_readiness",
        ],
        "review_doctrine": {
            "visual_first": True,
            "metadata_only_inference_forbidden": True,
            "all_pdf_pages_must_be_converted_before_review": True,
            "generated_figures_require_code_and_provenance_backtrace": True,
        },
        "target": {
            "figure_id": target.figure_id,
            "role": target.role,
            "asset": target.asset,
            "asset_layer": target.asset_layer,
            "provenance_mode": target.provenance_mode,
        },
        "outputs": {
            "role_reports_dir": "reviews",
            "final_review": "review.json",
        },
    }
    (bundle_dir / "workflow.json").write_text(json.dumps(workflow, indent=2, sort_keys=True), encoding="utf-8")


def _write_prompt(bundle_dir: Path, target: ReviewTarget, context_path: Path, bundle_hash: str) -> None:
    prompt = f"""# Codex Paper Asset Review

Review target: `{target.figure_id}`

Inputs:

- `preview.png`
- `preview_overlay.png`
- `{context_path.name}`
- `workflow.json`
- `reviews/visual-reviewer.template.json`
- `reviews/manuscript-fit-reviewer.template.json`
- `reviews/supervisor.template.json`

Use the `paper-asset-review` skill and the canonical Nature Communications requirements file from the context.

Required roles:

1. `visual-reviewer`
   - inspect the actual visual asset first; do not infer content from filenames, manuscript prose, or registry text
   - if any reviewed asset is a PDF, inspect the generated PNG previews for every page before drawing conclusions
   - inspect visual readability, hierarchy, spacing, overload, typography compliance, semantic color consistency, and narration restraint
   - write `reviews/visual-reviewer.json`
2. `manuscript-fit-reviewer`
   - inspect manuscript fit, claim support, caption delegation, and whether the asset belongs in the intended paper role
   - reconcile the figure's visual content with its generator or composition code and its evidence or provenance sources before concluding what the figure means
   - write `reviews/manuscript-fit-reviewer.json`
3. `supervisor`
   - reject reviews that skip visual inspection, PDF page conversion, or code/provenance reconciliation
   - consolidate both role reports
   - write final `review.json`

Asset model reminder:

- `context.json` distinguishes the manuscript-facing review asset from any upstream generator outputs and evidence sources.
- For multi-panel figures, inspect the split top-level panel assets listed in `context.json` to separate panel-local problems from recomposition problems.
- Use `registry_path`, `manuscript_path`, and the figure metadata in `context.json` to locate the generator or composition code path when needed.
- For `data_backed_*` provenance modes, judge the final manuscript asset as the release candidate, but use the upstream evidence references to detect provenance gaps or slide-style recomposition mistakes.
- If the final asset appears to discard or distort the data-backed upstream figure, call that out explicitly in the role reports.
- Apply the branch visual grammar: panel labels about 8 pt, most other figure text within 5–7 pt, restrained internal narration, and stable semantic colors across the paper.

Bundle hash:

- `{bundle_hash}`
"""
    (bundle_dir / "codex_review_prompt.md").write_text(prompt, encoding="utf-8")


def prepare_all(repo_root: Path | None = None) -> list[Path]:
    repo_root = _repo_root() if repo_root is None else repo_root
    review_dir, canonical_path, targets = _load_targets(repo_root)
    review_dir.mkdir(parents=True, exist_ok=True)
    canonical_rel = str(canonical_path.relative_to(repo_root))

    bundle_dirs: list[Path] = []
    for target in targets:
        asset_path = repo_root / target.asset
        bundle_dir = review_dir / target.figure_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        preview_path = bundle_dir / "preview.png"
        overlay_path = bundle_dir / "preview_overlay.png"
        context_path = bundle_dir / "context.json"

        if asset_path.suffix.lower() == ".pdf":
            _render_pdf_preview(asset_path, preview_path)
            pdf_pages_dir = bundle_dir / "preview_pages"
            asset_page_names = _render_pdf_page_previews(asset_path, pdf_pages_dir)
            asset_page_preview_paths = [str((pdf_pages_dir / name).relative_to(bundle_dir)) for name in asset_page_names]
        else:
            _render_image_preview(asset_path, preview_path)
            asset_page_preview_paths = [str(preview_path.relative_to(bundle_dir))]

        geometry = _geometry_for_asset(asset_path, _width_mm_for_mode(target.width_mode))
        layout_meta = _load_layout_metadata(repo_root, asset_path.stem)
        findings = _geometry_findings(target, geometry, layout_meta)
        panel_manifest = _load_panel_manifest(repo_root, target.panel_manifest)
        if len(target.panel_order) > 1 and panel_manifest is None:
            findings.append("Missing panel manifest for a multi-panel figure.")
        _draw_overlay(preview_path, overlay_path, geometry, layout_meta)
        panel_preview_records = _prepare_panel_previews(bundle_dir, repo_root, panel_manifest)

        context = {
            "figure_id": target.figure_id,
            "role": target.role,
            "asset": target.asset,
            "asset_layer": target.asset_layer,
            "provenance_mode": target.provenance_mode,
            "width_mode": target.width_mode,
            "manuscript_section": target.manuscript_section,
            "manuscript_path": target.manuscript_path,
            "manuscript_anchor": target.manuscript_anchor,
            "legend_path": target.legend_path,
            "registry_path": target.registry_path,
            "claim": target.claim,
            "panel_order": target.panel_order,
            "panel_manifest_path": target.panel_manifest,
            "legend_summary": target.legend_summary,
            "asset_model": {
                "manuscript_asset": target.asset,
                "asset_layer": target.asset_layer,
                "provenance_mode": target.provenance_mode,
                "split_panel_assets": panel_manifest,
                "evidence_sources": target.evidence_sources,
                "generator_outputs": target.generator_outputs,
                "provenance_note": target.provenance_note,
            },
            "canonical_requirements": canonical_rel,
            "inspection_requirements": {
                "visual_first": True,
                "metadata_only_inference_forbidden": True,
                "all_pdf_pages_must_be_converted_before_review": True,
                "generator_or_composition_code_must_be_inspected_for_generated_figures": True,
                "evidence_or_provenance_sources_must_be_inspected_for_generated_figures": True,
            },
            "asset_visual_review": {
                "asset_path": target.asset,
                "asset_suffix": asset_path.suffix.lower(),
                "page_preview_paths": asset_page_preview_paths,
            },
            "generator_trace_sources": {
                "registry_path": target.registry_path,
                "review_targets_path": "figures/conf/review_targets.yaml",
                "experiments_path": "figures/conf/experiments.yaml",
            },
            "geometry": geometry,
            "geometry_findings": findings,
            "layout_metadata_available": layout_meta is not None,
            "layout_metadata": layout_meta,
            "panel_previews": panel_preview_records,
        }
        context_path.write_text(json.dumps(context, indent=2, sort_keys=True), encoding="utf-8")

        bundle_hash = _compute_bundle_hash([preview_path, overlay_path, context_path])
        _write_role_templates(bundle_dir, bundle_hash, canonical_rel)
        _write_workflow(bundle_dir, target, canonical_rel, bundle_hash)
        _write_prompt(bundle_dir, target, context_path, bundle_hash)
        bundle_dirs.append(bundle_dir)

    print(f"Prepared {len(bundle_dirs)} Codex review bundle(s) in {review_dir}")
    return bundle_dirs


def gate_all(repo_root: Path | None = None) -> bool:
    repo_root = _repo_root() if repo_root is None else repo_root
    review_dir, _canonical_path, targets = _load_targets(repo_root)
    gate_rows: list[dict[str, Any]] = []
    all_ok = True

    for target in targets:
        bundle_dir = review_dir / target.figure_id
        context_path = bundle_dir / "context.json"
        review_path = bundle_dir / "review.json"
        issues: list[str] = []

        if not context_path.exists():
            issues.append("Missing context.json; run prepare first.")
        else:
            context = json.loads(context_path.read_text(encoding="utf-8"))
            issues.extend(context.get("geometry_findings", []))

        for role in REQUIRED_REVIEW_ROLES:
            role_report = bundle_dir / "reviews" / f"{role}.json"
            if not role_report.exists():
                issues.append(f"Missing reviews/{role}.json.")

        if not review_path.exists():
            issues.append("Missing review.json; the supervisor has not published a final decision.")
        else:
            review = json.loads(review_path.read_text(encoding="utf-8"))
            required_fields = (
                "skill_used",
                "canonical_requirements_path",
                "bundle_hash",
                "review_timestamp",
                "reviewers",
                "overall_verdict",
                "paper_role_fit",
                "primary_reason",
            )
            missing = [field for field in required_fields if field not in review]
            if missing:
                issues.append(f"review.json is missing required fields: {', '.join(missing)}.")
            roles_present = {
                item.get("reviewer_role")
                for item in review.get("reviewers", [])
                if isinstance(item, dict)
            }
            missing_roles = [role for role in REQUIRED_REVIEW_ROLES if role not in roles_present]
            if missing_roles:
                issues.append(f"review.json is missing reviewer roles: {', '.join(missing_roles)}.")
            if review.get("skill_used") != "paper-asset-review":
                issues.append("review.json does not declare skill_used=paper-asset-review.")
            if review.get("overall_verdict") != "pass":
                issues.append("Final review verdict is not pass.")
            fit = review.get("paper_role_fit")
            if fit not in ROLE_EXPECTATIONS.get(target.role, {target.role}):
                issues.append(f"Review says role fit is {fit!r}, expected one of {sorted(ROLE_EXPECTATIONS.get(target.role, {target.role}))}.")
            if review.get("split_recommended"):
                issues.append("Final review recommends splitting the figure before release.")
            if target.role == "main" and review.get("move_to_supplementary"):
                issues.append("Final review recommends moving this main figure to supplementary.")

        passed = not issues
        if not passed:
            all_ok = False
        gate_rows.append({"figure_id": target.figure_id, "role": target.role, "passed": passed, "issues": issues})

    report_path = review_dir / "gate_report.json"
    report_path.write_text(json.dumps(gate_rows, indent=2, sort_keys=True), encoding="utf-8")
    failed = [row for row in gate_rows if not row["passed"]]
    if failed:
        print(f"Codex review gate failed for {len(failed)} target(s). See {report_path}.")
    else:
        print(f"All {len(gate_rows)} review target(s) passed. See {report_path}.")
    return all_ok


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m figures.review {prepare|gate}")
        sys.exit(1)
    action = sys.argv[1]
    if action == "prepare":
        prepare_all()
        return
    if action == "gate":
        ok = gate_all()
        sys.exit(0 if ok else 1)
    print(f"Unknown action: {action}")
    sys.exit(1)


if __name__ == "__main__":
    main()
