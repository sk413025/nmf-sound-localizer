"""
Figure build pipeline — orchestrates generation, validation, and deployment.

Usage:
    python -m figures.build.pipeline generate
    python -m figures.build.pipeline validate
    python -m figures.build.pipeline deploy
    python -m figures.build.pipeline all
"""

from __future__ import annotations

import importlib
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

    print(f"Deployed {copied} files to {paper_dir}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m figures.build.pipeline {generate|validate|deploy|all}")
        sys.exit(1)

    action = sys.argv[1]

    if action in ("generate", "all"):
        cmd_generate()

    if action in ("validate", "all"):
        ok = cmd_validate()
        if not ok:
            print("\nValidation failed — deployment blocked.")
            sys.exit(1)

    if action in ("deploy", "all"):
        cmd_deploy()


if __name__ == "__main__":
    main()
