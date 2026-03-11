#!/usr/bin/env python3
"""Check that the manuscript contains the required top-level sections."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = REPO_ROOT / "paper" / "manuscript" / "manuscript.md"
REQUIRED_SECTIONS = [
    "Abstract",
    "Introduction",
    "Results",
    "Discussion",
    "Methods",
    "Data availability",
    "Code availability",
    "Acknowledgements",
    "References",
]


def main() -> int:
    if not MANUSCRIPT.exists():
        print(f"ERROR: missing manuscript source: {MANUSCRIPT}")
        return 1

    text = MANUSCRIPT.read_text(encoding="utf-8")
    headings = set(re.findall(r"^##\s+(.+)$", text, flags=re.MULTILINE))
    missing = [section for section in REQUIRED_SECTIONS if section not in headings]

    if missing:
        print("ERROR: manuscript is missing required sections:")
        for section in missing:
            print(f"- {section}")
        return 1

    print("OK: manuscript contains all required top-level sections.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
