#!/usr/bin/env python3
"""Check root-level asset boundaries and governance-safe routing."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "AGENTS.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "DATA_PROVENANCE.md",
    "LICENSE",
    "MANIFEST.in",
    "Makefile",
    "NATURE_FIGURE_GUIDELINES.md",
    "README.md",
    "START_HERE_AGENT.md",
    "START_HERE_HUMAN.md",
    "h_matrix_normalized_original_to_box.pth",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
}

ALLOWED_ROOT_DIRS = {
    ".codex",
    ".github",
    ".pytest_cache",
    "docs",
    "doa_rl",
    "figures",
    "legacy",
    "nmf_localizer",
    "paper",
    "results",
    "scripts",
    "tests",
}

FORBIDDEN_REFERENCE_PATHS = [
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "START_HERE_AGENT.md",
    REPO_ROOT / "START_HERE_HUMAN.md",
]

FORBIDDEN_REFERENCE_NEEDLES = [
    "docs/working-notes/",
    "legacy/",
]


def main() -> int:
    errors: list[str] = []

    for path in REPO_ROOT.iterdir():
        name = path.name
        if name == ".git":
            continue
        if path.is_dir():
            if name not in ALLOWED_ROOT_DIRS:
                errors.append(f"unclassified root directory: {name}")
        elif path.is_file():
            if name not in ALLOWED_ROOT_FILES:
                errors.append(f"unclassified root file: {name}")

    for path in FORBIDDEN_REFERENCE_PATHS:
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN_REFERENCE_NEEDLES:
            if needle in text:
                errors.append(f"{path.relative_to(REPO_ROOT)} must not route through non-canonical path: {needle}")

    if errors:
        print("ERROR: asset boundary check failed.")
        for error in errors:
            print(f"- {error}")
        return 1

    print("OK: root-level assets and canonical routing stay within declared boundaries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
