#!/usr/bin/env python3
"""Check manuscript-facing figure references against the registry and assets."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = REPO_ROOT / "paper" / "manuscript" / "manuscript.md"
REGISTRY = REPO_ROOT / "figures" / "FIGURE_REGISTRY.md"
PAPER_FIGURES = REPO_ROOT / "paper" / "figures"
MAIN_FIGURES = [1, 2, 3, 4, 5, 6]


def _check_exists(path: Path, label: str) -> list[str]:
    if path.exists():
        return []
    return [f"missing {label}: {path}"]


def main() -> int:
    errors: list[str] = []
    errors.extend(_check_exists(MANUSCRIPT, "manuscript"))
    errors.extend(_check_exists(REGISTRY, "figure registry"))
    errors.extend(_check_exists(PAPER_FIGURES, "paper figure directory"))
    if errors:
        print("ERROR:")
        for error in errors:
            print(f"- {error}")
        return 1

    manuscript_text = MANUSCRIPT.read_text(encoding="utf-8")
    registry_text = REGISTRY.read_text(encoding="utf-8")

    missing_refs: list[int] = []
    missing_registry: list[int] = []
    missing_assets: list[int] = []

    for fig_num in MAIN_FIGURES:
        if not re.search(rf"\bFig\.\s*{fig_num}\b", manuscript_text):
            missing_refs.append(fig_num)
        if f"### Fig {fig_num}" not in registry_text:
            missing_registry.append(fig_num)
        if not any(PAPER_FIGURES.glob(f"fig{fig_num:02d}_*")):
            missing_assets.append(fig_num)

    if missing_refs or missing_registry or missing_assets:
        print("ERROR: figure consistency check failed.")
        if missing_refs:
            print(f"- manuscript missing references for figures: {missing_refs}")
        if missing_registry:
            print(f"- figure registry missing entries for figures: {missing_registry}")
        if missing_assets:
            print(f"- paper/figures missing assets for figures: {missing_assets}")
        return 1

    print("OK: manuscript references, registry entries, and paper assets exist for Figs. 1-6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
