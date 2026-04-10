#!/usr/bin/env python3
"""Audit manuscript-facing self-diminishing phrasing and voice failures.

This script is a reviewer aid. It does not rewrite files or enforce a hard fail
on stylistic findings. It reports potential claim-floor suppressors and maps
them to the closest scientific-voice exemplar categories for rewrite guidance.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

TARGETS = [
    REPO_ROOT / "paper" / "manuscript" / "manuscript.md",
    REPO_ROOT / "paper" / "figures" / "Figure-Legends.md",
]

PATTERNS: list[tuple[str, str, bool, str]] = [
    ("without_upgrading", r"\bwithout upgrading\b", True, "SV3"),
    ("descriptive_rather_than", r"\bdescriptive rather than\b", True, "SV6"),
    ("remains_positive", r"\bremains positive\b", True, "SV5"),
    ("pathway_framing", r"\bpathway to\b", True, "SV1"),
    ("constraint_framing", r"\bcentral readout constraint\b|\bdefining the central constraint\b", True, "SV1"),
    ("rather_than_density", r"\brather than\b", False, "SV6"),
    ("remain_verb_density", r"\bremain(?:s|ed|ing)?\b", False, "SV5"),
]


def iter_lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            yield lineno, line.rstrip("\n")


def main() -> int:
    print("Scientific-voice audit")
    print("======================")
    print("This is a review aid, not a pass/fail gate.\n")

    grand_total = 0
    for path in TARGETS:
        print(path.relative_to(REPO_ROOT))
        local_total = 0
        for name, pattern, show_lines, exemplar in PATTERNS:
            regex = re.compile(pattern, re.IGNORECASE)
            hits = []
            for lineno, text in iter_lines(path):
                if regex.search(text):
                    hits.append((lineno, text.strip()))
            local_total += len(hits)
            if hits:
                print(f"  {name}: {len(hits)} (suggested exemplar: {exemplar})")
                if show_lines:
                    for lineno, text in hits:
                        print(f"    L{lineno}: {text}")
        if local_total == 0:
            print("  no flagged phrases found")
        print()
        grand_total += local_total

    print(f"Total flagged instances: {grand_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
