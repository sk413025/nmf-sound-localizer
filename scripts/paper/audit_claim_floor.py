#!/usr/bin/env python3
"""Audit paper-facing explanation for self-diminishing phrasing and sentence-friction risks.

This script is a reviewer aid. It does not rewrite files or enforce a hard fail
on stylistic findings. It reports potential claim-floor suppressors plus
sentence-friction hotspots across paper-facing explanation surfaces and maps
them to the closest scientific-voice exemplar categories for rewrite guidance.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

TARGETS = [
    REPO_ROOT / "paper" / "manuscript" / "manuscript.md",
    REPO_ROOT / "paper" / "manuscript" / "supplementary.md",
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
    ("static_verb_drag", r"\bstay(?:s|ed|ing)?\b|\bcontinue(?:s|d|ing)?\b", False, "SV13"),
    ("formal_register_trigger", r"\binterrogate\b|\boperationally costly\b", True, "SV12"),
]

WORD_RE = re.compile(r"[A-Za-z0-9°+-]+(?:-[A-Za-z0-9°+-]+)?")
COMPOUND_TRIGGER_RE = re.compile(
    r"\b("
    r"centered-magnitude|angle-frequency|clean-condition|five-seed|stage-0|"
    r"best-validation|full-matrix|per-angle|local-band|object-conditioned|"
    r"frequency-dependent|direction-selective|frequency-selective|half-plane|"
    r"moving-average|top-3|learned representation|readout problem"
    r")\b",
    re.IGNORECASE,
)
NUMBER_INTERPRET_TRIGGER_RE = re.compile(
    r"\b(from|to|rises?|drops?|jumps?|falls?|higher|lower|yet|still|agreement|aligns?|"
    r"accuracy|margin|burden|chance|gap|mean)\b|r\s*=",
    re.IGNORECASE,
)
CAUSAL_HINT_RE = re.compile(
    r"\b(because|which means|meaning that|confirming that|indicating that|therefore|so that|consistent with)\b",
    re.IGNORECASE,
)


def iter_lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            yield lineno, line.rstrip("\n")


def load_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def classify_surface(path: Path, lineno: int, lines: list[str]) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    if rel.endswith("Figure-Legends.md"):
        return "legends"
    if rel.endswith("supplementary.md"):
        return "supplementary"

    if lineno <= 8:
        return "main-manuscript salience"

    current_top = ""
    last_heading = 0
    paragraph_lines = 0
    in_paragraph = False
    for idx, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if stripped.startswith("## "):
            current_top = stripped[3:]
            last_heading = idx
            paragraph_lines = 0
            in_paragraph = False
        elif stripped.startswith("### "):
            last_heading = idx
            paragraph_lines = 0
            in_paragraph = False
        elif not stripped:
            paragraph_lines = 0
            in_paragraph = False
        else:
            if idx > last_heading:
                if not in_paragraph:
                    paragraph_lines = 1
                    in_paragraph = True
                else:
                    paragraph_lines += 1

        if idx == lineno:
            if current_top == "Methods":
                return "methods"
            if stripped.startswith("### "):
                return "main-manuscript salience"
            if current_top == "Discussion" and idx <= last_heading + 6:
                return "main-manuscript salience"
            if paragraph_lines and paragraph_lines <= 3 and idx > last_heading:
                return "main-manuscript salience"
            return "manuscript-body"

    return "manuscript-body"


def priority_for(surface: str, issue: str) -> str:
    if surface == "main-manuscript salience":
        return "rewrite-now"
    if surface in {"methods", "supplementary"} and issue in {
        "noun_stack_hotspot",
        "number_without_meaning",
        "static_verb_drag",
    }:
        return "technical-but-acceptable"
    if surface == "legends" and issue in {"noun_stack_hotspot", "number_without_meaning"}:
        return "watch"
    return "watch"


def noun_stack_hotspot(text: str) -> bool:
    compound_hits = len(COMPOUND_TRIGGER_RE.findall(text))
    if compound_hits >= 2 and len(WORD_RE.findall(text)) >= 10:
        return True
    return bool(
        re.search(
            r"\b(3-angle centered moving-average view|governed five-seed|"
            r"learned local neighborhood|measured local neighborhood|"
            r"five-seed mean P\(correct\)|best-validation checkpoints)\b",
            text,
            re.IGNORECASE,
        )
    )


def number_without_meaning(text: str) -> bool:
    numeric_markers = len(re.findall(r"\d", text))
    if numeric_markers < 3:
        return False
    if not NUMBER_INTERPRET_TRIGGER_RE.search(text):
        return False
    return CAUSAL_HINT_RE.search(text) is None


def main() -> int:
    print("Scientific-voice audit for paper-facing explanation")
    print("======================")
    print("This is a review aid, not a pass/fail gate.\n")

    grand_total = 0
    for path in TARGETS:
        print(path.relative_to(REPO_ROOT))
        lines = load_lines(path)
        local_total = 0
        for name, pattern, show_lines, exemplar in PATTERNS:
            regex = re.compile(pattern, re.IGNORECASE)
            hits = []
            for lineno, text in iter_lines(path):
                if regex.search(text):
                    surface = classify_surface(path, lineno, lines)
                    hits.append((lineno, text.strip(), surface, priority_for(surface, name)))
            local_total += len(hits)
            if hits:
                print(f"  {name}: {len(hits)} (suggested exemplar: {exemplar})")
                if show_lines:
                    for lineno, text, surface, priority in hits:
                        print(f"    [{priority}/{surface}] L{lineno}: {text}")
        noun_hits = []
        for lineno, text in iter_lines(path):
            if noun_stack_hotspot(text):
                surface = classify_surface(path, lineno, lines)
                noun_hits.append((lineno, text.strip(), surface, priority_for(surface, "noun_stack_hotspot")))
        local_total += len(noun_hits)
        if noun_hits:
            print(f"  noun_stack_hotspot: {len(noun_hits)} (suggested exemplar: SV9)")
            for lineno, text, surface, priority in noun_hits:
                print(f"    [{priority}/{surface}] L{lineno}: {text}")

        number_hits = []
        for lineno, text in iter_lines(path):
            if number_without_meaning(text):
                surface = classify_surface(path, lineno, lines)
                number_hits.append((lineno, text.strip(), surface, priority_for(surface, "number_without_meaning")))
        local_total += len(number_hits)
        if number_hits:
            print(
                "  number_without_meaning: "
                f"{len(number_hits)} (suggested exemplar: SV10/SV11)"
            )
            for lineno, text, surface, priority in number_hits:
                print(f"    [{priority}/{surface}] L{lineno}: {text}")
        if local_total == 0:
            print("  no flagged phrases found")
        print()
        grand_total += local_total

    print(f"Total flagged instances: {grand_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
