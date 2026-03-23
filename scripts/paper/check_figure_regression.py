#!/usr/bin/env python3
"""Compare active tracked figure artifacts against a baseline git ref."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "figures" / "conf" / "experiments.yaml"
ACTIVE_FIGURE_IDS = ("fig01", "fig02", "fig03", "fig04", "fig05", "fig06")


def _git(*args: str, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        input=input_bytes,
        capture_output=True,
        check=False,
    )


def _is_tracked(path: str) -> bool:
    result = _git("ls-files", "--error-unmatch", "--", path)
    return result.returncode == 0


def _looks_like_lfs_pointer(blob: bytes) -> bool:
    return blob.startswith(b"version https://git-lfs.github.com/spec/v1\n") and b"oid sha256:" in blob


def _resolve_git_blob(ref: str, path: str) -> bytes:
    result = _git("show", f"{ref}:{path}")
    if result.returncode != 0:
        raise FileNotFoundError(f"{path} not found at {ref}")
    blob = result.stdout
    if not _looks_like_lfs_pointer(blob):
        return blob
    smudged = _git("lfs", "smudge", input_bytes=blob)
    if smudged.returncode != 0:
        raise RuntimeError(f"git lfs smudge failed for {path} at {ref}: {smudged.stderr.decode('utf-8', errors='replace').strip()}")
    return smudged.stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _images_are_pixel_identical(current_path: Path, baseline_bytes: bytes) -> bool:
    with Image.open(current_path) as current_img:
        current_img = ImageOps.exif_transpose(current_img).convert("RGB")
        current_size = current_img.size
        current_pixels = current_img.tobytes()

    from io import BytesIO

    with Image.open(BytesIO(baseline_bytes)) as baseline_img:
        baseline_img = ImageOps.exif_transpose(baseline_img).convert("RGB")
        baseline_size = baseline_img.size
        baseline_pixels = baseline_img.tobytes()

    return current_size == baseline_size and current_pixels == baseline_pixels


def _normalize_figure_ids(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    requested = {item.strip().lower() for item in raw.split(",") if item.strip()}
    invalid = sorted(requested.difference(ACTIVE_FIGURE_IDS))
    if invalid:
        raise ValueError(f"Unsupported figure id(s): {', '.join(invalid)}")
    return requested


def _load_contract_paths(scope: str, figure_ids: set[str] | None = None) -> list[str]:
    with open(EXPERIMENTS, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    paths: list[str] = []
    for entry in cfg.values():
        if not isinstance(entry, dict) or not entry.get("figure_id"):
            continue
        if figure_ids is not None and entry["figure_id"] not in figure_ids:
            continue
        asset = entry.get("manuscript_asset")
        if asset:
            paths.append(asset)
            layout_rel = str(Path(asset).with_suffix(".layout.json"))
            if _is_tracked(layout_rel):
                paths.append(layout_rel)
        if scope == "paper":
            continue
        manifest = entry.get("panel_manifest")
        if manifest and _is_tracked(manifest):
            paths.append(manifest)
        for rel in entry.get("generator_outputs", []):
            if _is_tracked(rel):
                paths.append(rel)
    return sorted(dict.fromkeys(paths))


def _resolve_current_path(rel: str, paper_figures_dir: Path | None) -> Path:
    if paper_figures_dir is not None and rel.startswith("paper/figures/"):
        return paper_figures_dir / Path(rel).name
    return REPO_ROOT / rel


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare active tracked figure artifacts against a baseline git ref.")
    parser.add_argument("--baseline-ref", required=True, help="Git ref used as the regression baseline.")
    parser.add_argument(
        "--paper-figures-dir",
        help="Optional alternate directory containing staged paper/figures assets to compare against the baseline.",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "paper"),
        default="all",
        help="Whether to compare all tracked active figure artifacts or only manuscript paper assets.",
    )
    parser.add_argument(
        "--figures",
        help="Optional comma-separated active figure ids to compare (default: all active fig01-fig06).",
    )
    args = parser.parse_args()
    paper_figures_dir = Path(args.paper_figures_dir).resolve() if args.paper_figures_dir else None
    figure_ids = _normalize_figure_ids(args.figures)

    if not EXPERIMENTS.exists():
        print("ERROR: figures/conf/experiments.yaml not found", file=sys.stderr)
        return 1

    compared = 0
    mismatches: list[str] = []
    missing_current: list[str] = []
    for rel in _load_contract_paths(args.scope, figure_ids=figure_ids):
        current_path = _resolve_current_path(rel, paper_figures_dir)
        if not current_path.exists():
            missing_current.append(rel)
            continue
        current_bytes = current_path.read_bytes()
        try:
            baseline_bytes = _resolve_git_blob(args.baseline_ref, rel)
        except Exception as exc:
            mismatches.append(f"{rel}: {exc}")
            continue
        compared += 1
        if current_bytes != baseline_bytes:
            if args.scope == "paper" and current_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if _images_are_pixel_identical(current_path, baseline_bytes):
                    continue
            mismatches.append(
                f"{rel}: current {_sha256(current_bytes)[:12]} != {args.baseline_ref} {_sha256(baseline_bytes)[:12]}"
            )

    if missing_current:
        print("ERROR: current working tree is missing tracked regression artifacts:")
        for rel in missing_current:
            print(f"- {rel}")
        return 1

    if mismatches:
        print(f"FAIL: {len(mismatches)} regression mismatch(es) against {args.baseline_ref}")
        for line in mismatches:
            print(f"- {line}")
        return 1

    print(f"OK: {compared} active tracked {args.scope} figure artifact(s) match {args.baseline_ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
