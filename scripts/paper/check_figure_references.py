#!/usr/bin/env python3
"""Check active paper-facing figure references against the registry and asset contract."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = REPO_ROOT / "paper" / "manuscript" / "manuscript.md"
REGISTRY = REPO_ROOT / "figures" / "FIGURE_REGISTRY.md"
EXPERIMENTS = REPO_ROOT / "figures" / "conf" / "experiments.yaml"
PAPER_FIGURES = REPO_ROOT / "paper" / "figures"
FIGURE_DOCS = {"Figure-Legends.md", "README.md"}


def _check_exists(path: Path, label: str) -> list[str]:
    if path.exists():
        return []
    return [f"missing {label}: {path}"]


def _git_ls_files(*paths: str) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", *paths],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _load_active_contracts() -> dict[int, dict]:
    cfg = yaml.safe_load(EXPERIMENTS.read_text(encoding="utf-8")) or {}
    contracts: dict[int, dict] = {}
    for raw in cfg.values():
        if not isinstance(raw, dict):
            continue
        figure_id = raw.get("figure_id")
        manuscript_asset = raw.get("manuscript_asset")
        if not figure_id or not manuscript_asset:
            continue
        if not str(figure_id).startswith("fig"):
            continue
        fig_num = int(str(figure_id)[3:])
        if 1 <= fig_num <= 6:
            contracts[fig_num] = raw
    return contracts


def main() -> int:
    errors: list[str] = []
    errors.extend(_check_exists(MANUSCRIPT, "manuscript"))
    errors.extend(_check_exists(REGISTRY, "figure registry"))
    errors.extend(_check_exists(EXPERIMENTS, "figure contract"))
    errors.extend(_check_exists(PAPER_FIGURES, "paper figure directory"))
    if errors:
        print("ERROR:")
        for error in errors:
            print(f"- {error}")
        return 1

    contracts = _load_active_contracts()
    if sorted(contracts) != [1, 2, 3, 4, 5, 6]:
        print("ERROR: figures/conf/experiments.yaml does not define a complete active Fig. 1-6 manuscript contract.")
        print(f"- found active figures: {sorted(contracts)}")
        return 1

    manuscript_text = MANUSCRIPT.read_text(encoding="utf-8")
    registry_text = REGISTRY.read_text(encoding="utf-8")

    missing_refs: list[int] = []
    missing_registry: list[int] = []
    missing_assets: list[int] = []
    missing_layouts: list[str] = []

    expected_files = set(FIGURE_DOCS)
    tracked_paper_files = _git_ls_files("paper/figures")
    for tracked_rel in tracked_paper_files:
        tracked_name = Path(tracked_rel).name
        if tracked_name.endswith(".layout.json"):
            expected_files.add(tracked_name)

    for fig_num in sorted(contracts):
        asset_name = Path(contracts[fig_num]["manuscript_asset"]).name
        expected_files.add(asset_name)
        layout_name = Path(asset_name).with_suffix(".layout.json").name
        if (PAPER_FIGURES / layout_name).exists():
            expected_files.add(layout_name)
        if not re.search(rf"\bFig\.\s*{fig_num}\b", manuscript_text):
            missing_refs.append(fig_num)
        if f"### Fig {fig_num} " not in registry_text and f"### Fig {fig_num} —" not in registry_text:
            missing_registry.append(fig_num)
        if not (PAPER_FIGURES / asset_name).exists():
            missing_assets.append(fig_num)

    for expected_name in sorted(expected_files):
        if not (PAPER_FIGURES / expected_name).exists():
            if expected_name.endswith(".layout.json"):
                missing_layouts.append(expected_name)

    actual_files = {path.name for path in PAPER_FIGURES.iterdir() if path.is_file()}
    unexpected_files = sorted(actual_files - expected_files)

    if missing_refs or missing_registry or missing_assets or missing_layouts or unexpected_files:
        print("ERROR: figure consistency check failed.")
        if missing_refs:
            print(f"- manuscript missing references for figures: {missing_refs}")
        if missing_registry:
            print(f"- figure registry missing entries for figures: {missing_registry}")
        if missing_assets:
            print(f"- paper/figures missing canonical paper-facing assets for figures: {missing_assets}")
        if missing_layouts:
            print(f"- paper/figures missing tracked layout sidecars: {missing_layouts}")
        if unexpected_files:
            print(f"- paper/figures contains non-canonical files: {unexpected_files}")
        return 1

    print("OK: main-manuscript references, registry entries, and the canonical paper/figures paper-facing asset surface match the active Fig. 1-6 contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
