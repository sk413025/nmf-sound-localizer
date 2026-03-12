"""
Figure build pipeline — orchestrates generation, validation, and deployment.

Usage:
    python -m figures.build.pipeline generate
    python -m figures.build.pipeline validate
    python -m figures.build.pipeline review_prepare
    python -m figures.build.pipeline review_gate
    python -m figures.build.pipeline deploy
    python -m figures.build.pipeline release
    python -m figures.build.pipeline all
"""

from __future__ import annotations

import importlib
import subprocess
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _discover_generators() -> list:
    """Dynamically discover all generator modules in figures/generators/."""
    generators_dir = Path(__file__).resolve().parent.parent / "generators"
    modules = []
    for py_file in sorted(generators_dir.glob("fig*.py")):
        module_name = f"figures.generators.{py_file.stem}"
        mod = importlib.import_module(module_name)
        if hasattr(mod, "generate"):
            modules.append(mod)
    return modules


def cmd_generate() -> list[Path]:
    """Run all generators. Returns list of output paths."""
    from figures.style import load_paths
    from figures.panel_assets import prepare_all as prepare_panel_assets

    root = _repo_root()
    paths_cfg = load_paths()
    output_dir = root / paths_cfg.get("output_dir", "figures/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_outputs: list[Path] = []
    generators = _discover_generators()
    print(f"Discovered {len(generators)} generators")

    for gen in generators:
        name = gen.__name__.split(".")[-1]
        print(f"\n--- {name} ---")
        try:
            outputs = gen.generate(data_root=root, output_dir=output_dir)
            all_outputs.extend(outputs)
        except Exception as e:
            print(f"[{name}] ERROR: {e}")

    all_outputs.extend(prepare_panel_assets(root))
    print(f"\nGeneration complete: {len(all_outputs)} files")
    return all_outputs


def cmd_validate() -> bool:
    """Validate all outputs. Returns True if all pass."""
    from figures.style import load_paths
    from figures.validate import validate_all

    root = _repo_root()
    paths_cfg = load_paths()
    output_dir = root / paths_cfg.get("output_dir", "figures/output")

    if not output_dir.exists() or not list(output_dir.glob("*.pdf")):
        print(f"No outputs to validate in {output_dir}")
        return True

    return validate_all(output_dir)


def cmd_deploy() -> None:
    """Copy validated outputs to paper/figures/."""
    from figures.style import load_paths

    root = _repo_root()
    paths_cfg = load_paths()
    output_dir = root / paths_cfg.get("output_dir", "figures/output")
    paper_dir = root / paths_cfg.get("paper_figures_dir", "paper/figures")
    paper_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(output_dir.glob("*.pdf"))
    tiff_files = sorted(output_dir.glob("*.tiff"))

    copied = 0
    for f in pdf_files + tiff_files:
        dest = paper_dir / f.name
        shutil.copy2(f, dest)
        copied += 1

    compose_script = root / "scripts" / "paper" / "compose_master_figure3_family.py"
    if compose_script.exists():
        subprocess.run(
            [sys.executable, str(compose_script)],
            check=True,
            cwd=str(root),
        )

    print(f"Deployed {copied} files to {paper_dir}")


def cmd_review_prepare() -> list[Path]:
    """Build Codex review bundles for paper-facing figure assets."""
    from figures.review import prepare_all

    return prepare_all(_repo_root())


def cmd_review_gate() -> bool:
    """Require passing Codex review reports before release deployment."""
    from figures.review import gate_all

    return gate_all(_repo_root())


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m figures.build.pipeline {generate|validate|review_prepare|review_gate|deploy|release|all}")
        sys.exit(1)

    action = sys.argv[1]

    if action in ("generate", "all"):
        cmd_generate()

    if action in ("validate", "all"):
        ok = cmd_validate()
        if not ok:
            print("\nValidation failed — deployment blocked.")
            sys.exit(1)

    if action in ("review_prepare", "all", "release"):
        cmd_review_prepare()

    if action in ("review_gate", "release"):
        ok = cmd_review_gate()
        if not ok:
            print("\nCodex review gate failed — release blocked.")
            sys.exit(1)

    if action in ("deploy", "all", "release"):
        cmd_deploy()


if __name__ == "__main__":
    main()
