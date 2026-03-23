#!/usr/bin/env python3
"""Rebuild the active six manuscript figures from the unified figure contract."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "figures" / "conf" / "experiments.yaml"
DEFAULT_STAGE_DIR = REPO_ROOT / "results" / "paper_active_figure_rerun"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_active_contracts() -> list[dict[str, Any]]:
    with open(EXPERIMENTS, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    contracts: list[dict[str, Any]] = []
    for raw in cfg.values():
        if not isinstance(raw, dict):
            continue
        figure_id = raw.get("figure_id")
        manuscript_asset = raw.get("manuscript_asset")
        generator = raw.get("generator")
        if not figure_id or not manuscript_asset or not generator:
            continue
        number = int(str(figure_id)[3:])
        if 1 <= number <= 6:
            contracts.append(raw)
    contracts.sort(key=lambda item: int(str(item["figure_id"])[3:]))
    return contracts


def _run(args: list[str]) -> None:
    subprocess.run(args, cwd=str(REPO_ROOT), check=True)


def _stage_paper_assets(contracts: list[dict[str, Any]], stage_paper_dir: Path, promote: bool) -> None:
    live_paper_dir = REPO_ROOT / "paper" / "figures"
    target_root = live_paper_dir if promote else stage_paper_dir
    target_root.mkdir(parents=True, exist_ok=True)

    for contract in contracts:
        rel_asset = Path(contract["manuscript_asset"])
        staged_asset = stage_paper_dir / rel_asset.name
        if not staged_asset.exists():
            raise FileNotFoundError(f"Missing staged manuscript asset: {staged_asset}")

        live_asset = REPO_ROOT / rel_asset
        destination = live_asset if promote else staged_asset
        if promote:
            shutil.copy2(staged_asset, destination)

        stage_layout = staged_asset.with_suffix(".layout.json")
        live_layout = live_asset.with_suffix(".layout.json")
        if stage_layout.exists() and promote:
            shutil.copy2(stage_layout, live_layout)

    if promote:
        print(f"Promoted staged manuscript assets into {live_paper_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the active six manuscript figures under the unified figure contract.")
    parser.add_argument(
        "--baseline-ref",
        default="HEAD",
        help="Git ref used as the regression baseline for staged paper assets.",
    )
    parser.add_argument(
        "--stage-dir",
        default=str(DEFAULT_STAGE_DIR),
        help="Directory used to stage regenerated paper-facing assets before promotion.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Copy staged paper-facing assets into paper/figures after regression passes.",
    )
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir).resolve()
    stage_paper_dir = stage_dir / "paper_figures"

    contracts = _load_active_contracts()
    if not contracts:
        print("ERROR: no active fig01-fig06 contracts found in figures/conf/experiments.yaml", file=sys.stderr)
        return 1

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_paper_dir.mkdir(parents=True, exist_ok=True)

    from figures.build.pipeline import cmd_generate, cmd_validate

    print("Generating active figure outputs into figures/output ...")
    cmd_generate()

    print("\nValidating active generator outputs ...")
    if not cmd_validate():
        return 1

    print(f"\nComposing manuscript assets into staging directory: {stage_paper_dir}")
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "paper" / "compose_master_figure3_family.py"),
            "--paper-dir",
            str(stage_paper_dir),
        ]
    )

    print(f"\nComparing staged paper assets against baseline {args.baseline_ref} ...")
    regression = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "paper" / "check_figure_regression.py"),
            "--baseline-ref",
            args.baseline_ref,
            "--paper-figures-dir",
            str(stage_paper_dir),
            "--scope",
            "paper",
        ],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if regression.returncode != 0:
        print("\nRegression check failed; staged assets were kept for inspection and not promoted.", file=sys.stderr)
        return regression.returncode

    if args.promote:
        _stage_paper_assets(contracts, stage_paper_dir, promote=True)
    else:
        _stage_paper_assets(contracts, stage_paper_dir, promote=False)
        print("\nStage complete; rerun with --promote to overwrite paper/figures after review.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
