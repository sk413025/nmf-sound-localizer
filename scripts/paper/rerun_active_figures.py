#!/usr/bin/env python3
"""Rebuild the active six paper-facing figures from the unified figure contract."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "figures" / "conf" / "experiments.yaml"
DEFAULT_STAGE_DIR = REPO_ROOT / "results" / "paper_active_figure_rerun"
ACTIVE_FIGURE_IDS = ("fig01", "fig02", "fig03", "fig04", "fig05", "fig06")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.layout_contract import contract_version, figure_typography, source_layout_spec


def _normalize_figure_ids(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    requested = {item.strip().lower() for item in raw.split(",") if item.strip()}
    invalid = sorted(requested.difference(ACTIVE_FIGURE_IDS))
    if invalid:
        raise ValueError(f"Unsupported figure id(s): {', '.join(invalid)}")
    return requested


def _load_active_contracts(figure_ids: set[str] | None = None) -> list[dict[str, Any]]:
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
            if figure_ids is not None and figure_id not in figure_ids:
                continue
            contracts.append(raw)
    contracts.sort(key=lambda item: int(str(item["figure_id"])[3:]))
    return contracts


def _run(args: list[str]) -> None:
    subprocess.run(args, cwd=str(REPO_ROOT), check=True)


def _write_mixed_panel_manifest(contract: dict[str, Any]) -> None:
    manifest_rel = contract.get("panel_manifest")
    panel_map = contract.get("panel_provenance")
    if not manifest_rel or not isinstance(panel_map, dict):
        return
    if "manuscript_panels" not in str(manifest_rel):
        return

    panel_order = list(contract.get("panel_order", []))
    panels: list[dict[str, Any]] = []
    for panel_id in panel_order:
        panel = panel_map.get(panel_id)
        if not isinstance(panel, dict):
            raise KeyError(
                f"{contract.get('figure_id', 'unknown')}: missing panel_provenance entry for panel {panel_id!r}"
            )
        asset_path = panel.get("asset_path")
        if not asset_path:
            raise KeyError(
                f"{contract.get('figure_id', 'unknown')}.{panel_id}: missing asset_path in panel_provenance"
            )
        panels.append(
            {
                "asset_path": asset_path,
                "panel_id": panel_id,
                "provenance_mode": panel.get("provenance_mode"),
                "source_asset": asset_path,
                "source_layer": "mixed_panel_lineage",
                "storage_mode": "reference_existing_outputs",
                "title": panel.get("title"),
            }
        )

    manifest_path = REPO_ROOT / manifest_rel
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "composite_asset": contract["manuscript_asset"],
        "contract_version": contract_version(),
        "figure_id": contract["figure_id"],
        "panel_order": panel_order,
        "panels": panels,
        "source_layout_spec": source_layout_spec(),
        "storage_mode": "reference_existing_outputs",
        "typography_pt": figure_typography(contract["figure_id"]),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _snapshot_generator_outputs(contracts: list[dict[str, Any]]) -> dict[str, bytes | None]:
    snapshot: dict[str, bytes | None] = {}
    for contract in contracts:
        for rel in contract.get("generator_outputs", []):
            path = REPO_ROOT / rel
            snapshot[rel] = path.read_bytes() if path.exists() else None
    return snapshot


def _changed_generator_outputs(
    before: dict[str, bytes | None], contracts: list[dict[str, Any]]
) -> dict[str, list[str]]:
    changed_by_figure: dict[str, list[str]] = {}
    for contract in contracts:
        figure_id = contract["figure_id"]
        changed: list[str] = []
        for rel in contract.get("generator_outputs", []):
            path = REPO_ROOT / rel
            current = path.read_bytes() if path.exists() else None
            if current != before.get(rel):
                changed.append(rel)
        if changed:
            changed_by_figure[figure_id] = changed
    return changed_by_figure


def _images_are_pixel_identical(path_a: Path, path_b: Path) -> bool:
    with Image.open(path_a) as img_a:
        img_a = ImageOps.exif_transpose(img_a).convert("RGB")
        size_a = img_a.size
        pixels_a = img_a.tobytes()
    with Image.open(path_b) as img_b:
        img_b = ImageOps.exif_transpose(img_b).convert("RGB")
        size_b = img_b.size
        pixels_b = img_b.tobytes()
    return size_a == size_b and pixels_a == pixels_b


def _warn_if_staged_paper_asset_unchanged(
    contracts: list[dict[str, Any]],
    stage_paper_dir: Path,
    changed_outputs: dict[str, list[str]],
) -> None:
    for contract in contracts:
        figure_id = contract["figure_id"]
        if figure_id not in changed_outputs:
            continue
        rel_asset = Path(contract["manuscript_asset"])
        staged_asset = stage_paper_dir / rel_asset.name
        live_asset = REPO_ROOT / rel_asset
        if not staged_asset.exists() or not live_asset.exists():
            continue
        if staged_asset.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if _images_are_pixel_identical(staged_asset, live_asset):
            changed = ", ".join(changed_outputs[figure_id])
            print(
                (
                    f"WARNING: {figure_id} generator outputs changed during this run "
                    f"({changed}), but the staged paper asset {staged_asset.name} is still "
                    "pixel-identical to the live paper asset. Confirm compose ordering or inspect the staged output."
                ),
                file=sys.stderr,
            )


def _stage_paper_assets(contracts: list[dict[str, Any]], stage_paper_dir: Path, promote: bool) -> None:
    live_paper_dir = REPO_ROOT / "paper" / "figures"
    target_root = live_paper_dir if promote else stage_paper_dir
    target_root.mkdir(parents=True, exist_ok=True)

    for contract in contracts:
        rel_asset = Path(contract["manuscript_asset"])
        staged_asset = stage_paper_dir / rel_asset.name
        if not staged_asset.exists():
            raise FileNotFoundError(f"Missing staged paper-facing asset: {staged_asset}")

        live_asset = REPO_ROOT / rel_asset
        destination = live_asset if promote else staged_asset
        if promote:
            shutil.copy2(staged_asset, destination)

        stage_layout = staged_asset.with_suffix(".layout.json")
        live_layout = live_asset.with_suffix(".layout.json")
        if stage_layout.exists() and promote:
            shutil.copy2(stage_layout, live_layout)

    if promote:
        print(f"Promoted staged paper-facing assets into {live_paper_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the active six paper-facing figures under the unified figure contract.")
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
    parser.add_argument(
        "--figures",
        help="Optional comma-separated active figure ids to rebuild (default: all active fig01-fig06).",
    )
    parser.add_argument(
        "--allow-paper-diff",
        action="store_true",
        help="Allow promotion after review even if staged paper assets differ from --baseline-ref.",
    )
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir).resolve()
    stage_paper_dir = stage_dir / "paper_figures"
    figure_ids = _normalize_figure_ids(args.figures)

    contracts = _load_active_contracts(figure_ids=figure_ids)
    if not contracts:
        print("ERROR: no active fig01-fig06 contracts found in figures/conf/experiments.yaml", file=sys.stderr)
        return 1

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_paper_dir.mkdir(parents=True, exist_ok=True)

    from figures.build.pipeline import cmd_generate, cmd_validate

    before_outputs = _snapshot_generator_outputs(contracts)

    print("Generating active figure outputs into figures/output ...")
    cmd_generate(figure_ids={contract["figure_id"] for contract in contracts})

    print("\nValidating active generator outputs ...")
    if not cmd_validate(figure_ids={contract["figure_id"] for contract in contracts}):
        return 1

    changed_outputs = _changed_generator_outputs(before_outputs, contracts)

    print(f"\nComposing manuscript assets into staging directory: {stage_paper_dir}")
    selected_ids = ",".join(contract["figure_id"] for contract in contracts)
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "paper" / "compose_master_figure3_family.py"),
            "--paper-dir",
            str(stage_paper_dir),
            "--figures",
            selected_ids,
        ]
    )

    for contract in contracts:
        _write_mixed_panel_manifest(contract)

    _warn_if_staged_paper_asset_unchanged(contracts, stage_paper_dir, changed_outputs)

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
            "--figures",
            selected_ids,
        ],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if regression.returncode != 0:
        if args.promote and args.allow_paper_diff:
            print(
                (
                    "\nRegression check reported a paper-asset diff against the baseline, "
                    "but --allow-paper-diff was set; proceeding with promotion for the selected figure(s)."
                ),
                file=sys.stderr,
            )
        else:
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
