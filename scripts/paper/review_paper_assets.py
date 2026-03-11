#!/usr/bin/env python3
"""Manuscript-first entrypoint for Codex-governed paper asset review."""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _usage() -> None:
    print("Usage: python scripts/paper/review_paper_assets.py {prepare|gate}")


def main() -> None:
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)

    action = sys.argv[1]
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))

    from figures.review import gate_all, prepare_all

    if action == "prepare":
        prepare_all(repo_root)
        return
    if action == "gate":
        ok = gate_all(repo_root)
        sys.exit(0 if ok else 1)

    _usage()
    sys.exit(1)


if __name__ == "__main__":
    main()
