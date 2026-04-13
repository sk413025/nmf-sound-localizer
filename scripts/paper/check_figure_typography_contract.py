#!/usr/bin/env python3
"""Check that active Fig. 1–6 typography comes from the layout contract."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.layout_contract import (
    contract_version,
    figure_typography,
    source_layout_spec,
)

ACTIVE_CODE_PATHS = [
    REPO_ROOT / "figures/generators/fig01_paradigm_data.py",
    REPO_ROOT / "figures/generators/fig02_svd_spectrum.py",
    REPO_ROOT / "figures/generators/fig03_fingerprint_discriminability.py",
    REPO_ROOT / "figures/generators/fig04_solver_dynamics.py",
    REPO_ROOT / "figures/generators/fig05_performance_structure.py",
    REPO_ROOT / "figures/generators/fig06_universality.py",
    REPO_ROOT / "scripts/paper/compose_master_figure3_family.py",
]
ACTIVE_METADATA_PATHS = [
    REPO_ROOT / "paper/figures/fig01_paradigm-shift.layout.json",
    REPO_ROOT / "paper/figures/fig02_svd-physical-dictionary.layout.json",
    REPO_ROOT / "paper/figures/fig03_fingerprint-discriminability.layout.json",
    REPO_ROOT / "paper/figures/fig04_solver-dynamics.layout.json",
    REPO_ROOT / "paper/figures/fig05_performance-structure.layout.json",
    REPO_ROOT / "paper/figures/fig06_universality.layout.json",
    REPO_ROOT / "figures/output/fig01_paradigm_shift_panels/fig01_panel_manifest.json",
    REPO_ROOT / "figures/output/fig02_svd_spectrum_panels/fig02_panel_manifest.json",
    REPO_ROOT / "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_manifest.json",
    REPO_ROOT / "figures/output/fig04_solver_dynamics_manuscript_panels/fig04_panel_manifest.json",
    REPO_ROOT / "figures/output/fig05_performance_structure_panels/fig05_panel_manifest.json",
    REPO_ROOT / "figures/output/fig06_universality_manuscript_panels/fig06_panel_manifest.json",
]
ACTIVE_SOURCE_PDFS = [
    REPO_ROOT / "figures/output/fig01_paradigm_data.pdf",
    REPO_ROOT / "figures/output/fig02_svd_spectrum.pdf",
    REPO_ROOT / "figures/output/fig03_fingerprint_discriminability.pdf",
    REPO_ROOT / "figures/output/fig04_solver_dynamics.pdf",
    REPO_ROOT / "figures/output/fig05_performance_structure.pdf",
    REPO_ROOT / "figures/output/fig06_universality.pdf",
]
FORBIDDEN_PATTERNS = {
    "hardcoded_panel_label_px": re.compile(r"\bPANEL_LABEL_PX\s*="),
    "hardcoded_font_constant": re.compile(
        r"\b(?:FIG0[1-6])_[A-Z_]*?(?:PT|FONTSIZE)\s*=\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    ),
    "numeric_fontsize_literal": re.compile(r"\bfontsize\b\s*=\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)"),
    "numeric_labelsize_literal": re.compile(r"\blabelsize\b\s*=\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)"),
}

FIGURE_ID_PATTERN = re.compile(r"(fig0[1-6])")
ALLOWED_FONT_FAMILIES = ("Arial", "Helvetica")


def _check_code() -> list[str]:
    issues: list[str] = []
    for path in ACTIVE_CODE_PATHS:
        text = path.read_text(encoding="utf-8")
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                issues.append(f"{path.relative_to(REPO_ROOT)} still contains {name}")
    return issues


def _check_metadata() -> list[str]:
    expected_version = contract_version()
    expected_source = source_layout_spec()
    issues: list[str] = []

    for path in ACTIVE_METADATA_PATHS:
        if not path.exists():
            issues.append(f"missing metadata file: {path.relative_to(REPO_ROOT)}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("contract_version") != expected_version:
            issues.append(
                f"{path.relative_to(REPO_ROOT)} has contract_version={payload.get('contract_version')} expected {expected_version}"
            )
        if payload.get("source_layout_spec") != expected_source:
            issues.append(
                f"{path.relative_to(REPO_ROOT)} has source_layout_spec={payload.get('source_layout_spec')} expected {expected_source}"
            )
        figure_match = FIGURE_ID_PATTERN.search(path.name)
        if figure_match is None:
            figure_match = FIGURE_ID_PATTERN.search(str(path.relative_to(REPO_ROOT)))
        if figure_match is None:
            issues.append(f"could not infer figure id for metadata file: {path.relative_to(REPO_ROOT)}")
            continue
        expected_fonts = figure_typography(figure_match.group(1))
        if payload.get("typography_pt") != expected_fonts:
            issues.append(f"{path.relative_to(REPO_ROOT)} typography_pt does not match current layout contract")
    return issues


def _clean_font_name(font_name: str) -> str:
    base_name = font_name.split("+")[-1] if "+" in font_name else font_name
    return base_name.split(",")[0].strip()


def _check_source_pdf_fonts() -> list[str]:
    issues: list[str] = []
    for path in ACTIVE_SOURCE_PDFS:
        if not path.exists():
            issues.append(f"missing source pdf: {path.relative_to(REPO_ROOT)}")
            continue
        doc = fitz.open(str(path))
        try:
            for page in doc:
                for font_entry in page.get_fonts(full=True):
                    font_name = _clean_font_name(font_entry[3] if len(font_entry) > 3 else "")
                    if not font_name:
                        continue
                    if any(allowed.lower() in font_name.lower() for allowed in ALLOWED_FONT_FAMILIES):
                        continue
                    issues.append(
                        f"{path.relative_to(REPO_ROOT)} embeds non-submission font {font_name!r}"
                    )
        finally:
            doc.close()
    return issues


def main() -> int:
    issues = _check_code() + _check_metadata() + _check_source_pdf_fonts()
    if issues:
        print("FAIL: active Fig. 1–6 typography contract is not fully enforced")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("OK: active Fig. 1–6 typography is sourced from layout_spec.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
