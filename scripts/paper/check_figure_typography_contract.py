#!/usr/bin/env python3
"""Check that active Fig. 4–6 typography comes from the layout contract."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.layout_contract import contract_version, font_tokens, source_layout_spec

ACTIVE_CODE_PATHS = [
    REPO_ROOT / "figures/generators/fig04_solver_dynamics.py",
    REPO_ROOT / "figures/generators/fig05_performance_structure.py",
    REPO_ROOT / "figures/generators/fig06_universality.py",
    REPO_ROOT / "scripts/paper/compose_master_figure3_family.py",
]
ACTIVE_METADATA_PATHS = [
    REPO_ROOT / "paper/figures/fig04_solver-dynamics.layout.json",
    REPO_ROOT / "paper/figures/fig06_universality.layout.json",
    REPO_ROOT / "figures/output/fig04_solver_dynamics_panels/fig04_panel_manifest.json",
    REPO_ROOT / "figures/output/fig05_performance_structure_panels/fig05_panel_manifest.json",
    REPO_ROOT / "figures/output/fig06_universality_panels/fig06_panel_manifest.json",
]
FORBIDDEN_PATTERNS = {
    "hardcoded_panel_label_px": re.compile(r"\bPANEL_LABEL_PX\s*="),
    "hardcoded_font_constant": re.compile(
        r"\b(?:FIG04|FIG05|FIG06)_[A-Z_]*?(?:PT|FONTSIZE)\s*=\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    ),
    "numeric_fontsize_literal": re.compile(r"fontsize\s*=\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)"),
}


def _check_code() -> list[str]:
    issues: list[str] = []
    for path in ACTIVE_CODE_PATHS:
        text = path.read_text(encoding="utf-8")
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                issues.append(f"{path.relative_to(REPO_ROOT)} still contains {name}")
    return issues


def _check_metadata() -> list[str]:
    expected_fonts = font_tokens()
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
        if payload.get("typography_pt") != expected_fonts:
            issues.append(f"{path.relative_to(REPO_ROOT)} typography_pt does not match current layout contract")
    return issues


def main() -> int:
    issues = _check_code() + _check_metadata()
    if issues:
        print("FAIL: active Fig. 4–6 typography contract is not fully enforced")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("OK: active Fig. 4–6 typography is sourced from layout_spec.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
