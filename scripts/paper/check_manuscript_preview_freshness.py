#!/usr/bin/env python3
"""Check whether manuscript preview outputs are fresher than referenced figures."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_MD = REPO_ROOT / "paper" / "manuscript" / "manuscript.md"
PREVIEW_OUTPUTS = [
    REPO_ROOT / "paper" / "out" / "manuscript.docx",
    REPO_ROOT / "paper" / "out" / "manuscript.pdf",
]
FIGURE_PATTERN = re.compile(r"\.\./figures/([A-Za-z0-9_.-]+\.(?:jpg|jpeg|png))")


def _referenced_figure_paths() -> list[Path]:
    text = MANUSCRIPT_MD.read_text(encoding="utf-8")
    matches = sorted(set(FIGURE_PATTERN.findall(text)))
    return [REPO_ROOT / "paper" / "figures" / name for name in matches]


def main() -> int:
    figures = _referenced_figure_paths()
    missing_figures = [path for path in figures if not path.exists()]
    if missing_figures:
        print("FAIL: manuscript references missing paper-facing figure assets")
        for path in missing_figures:
            print(f"- missing figure asset: {path.relative_to(REPO_ROOT)}")
        return 1

    issues: list[str] = []
    latest_figure_mtime = max(path.stat().st_mtime for path in figures)
    for output in PREVIEW_OUTPUTS:
        if not output.exists():
            continue
        if output.stat().st_mtime < latest_figure_mtime:
            issues.append(
                f"{output.relative_to(REPO_ROOT)} is older than at least one referenced paper/figures asset"
            )

    if issues:
        print("FAIL: manuscript preview outputs are stale relative to paper-facing figure assets")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("OK: manuscript preview outputs are fresh relative to referenced paper-facing figure assets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
