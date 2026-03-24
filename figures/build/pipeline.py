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
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_active_contracts(repo_root: Path | None = None, figure_ids: set[str] | None = None) -> list[dict[str, Any]]:
    repo_root = repo_root or _repo_root()
    cfg_path = repo_root / "figures" / "conf" / "experiments.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    contracts: list[dict[str, Any]] = []
    for raw in cfg.values():
        if not isinstance(raw, dict):
            continue
        figure_id = raw.get("figure_id")
        generator = raw.get("generator")
        if not figure_id or not generator:
            continue
        if not figure_id.startswith("fig"):
            continue
        number = int(figure_id[3:])
        if 1 <= number <= 6:
            if figure_ids is not None and figure_id not in figure_ids:
                continue
            contracts.append(raw)
    contracts.sort(key=lambda item: int(str(item["figure_id"])[3:]))
    return contracts


def _import_generator(generator_path: str):
    module_name = generator_path.replace("/", ".")
    if module_name.endswith(".py"):
        module_name = module_name[:-3]
    module = importlib.import_module(module_name)
    if not hasattr(module, "generate"):
        raise AttributeError(f"{generator_path} has no generate() entrypoint")
    return module


def _active_generators(contracts: list[dict[str, Any]]) -> list:
    modules = []
    seen: set[str] = set()
    for contract in contracts:
        generator_path = contract["generator"]
        if generator_path in seen:
            continue
        seen.add(generator_path)
        modules.append(_import_generator(generator_path))
    return modules


def _declared_generator_outputs(repo_root: Path, contracts: list[dict[str, Any]]) -> list[Path]:
    outputs: list[Path] = []
    seen: set[Path] = set()
    for contract in contracts:
        for rel in contract.get("generator_outputs", []):
            path = repo_root / rel
            if path in seen:
                continue
            seen.add(path)
            outputs.append(path)
    return outputs


def cmd_generate(figure_ids: set[str] | None = None) -> list[Path]:
    """Run the active six-figure generators declared in experiments.yaml."""
    from figures.style import load_paths

    root = _repo_root()
    paths_cfg = load_paths()
    output_dir = root / paths_cfg.get("output_dir", "figures/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_outputs: list[Path] = []
    contracts = _load_active_contracts(root, figure_ids=figure_ids)
    generators = _active_generators(contracts)
    target_msg = "all active figures" if figure_ids is None else ", ".join(sorted(figure_ids))
    print(f"Running {len(generators)} active generators from experiments.yaml for {target_msg}")

    for gen in generators:
        name = gen.__name__.split(".")[-1]
        print(f"\n--- {name} ---")
        outputs = gen.generate(data_root=root, output_dir=output_dir)
        all_outputs.extend(outputs)

    print(f"\nGeneration complete: {len(all_outputs)} files")
    return all_outputs


def cmd_validate(figure_ids: set[str] | None = None) -> bool:
    """Validate active generator outputs declared in experiments.yaml."""
    from figures.validate import validate_file

    root = _repo_root()
    contracts = _load_active_contracts(root, figure_ids=figure_ids)
    outputs = _declared_generator_outputs(root, contracts)
    all_paths = set(outputs)

    if not outputs:
        print("No declared generator outputs to validate")
        return True

    missing = [path for path in outputs if not path.exists()]
    if missing:
        print("ERROR: declared generator outputs are missing:")
        for path in missing:
            print(f"- {path}")
        return False

    reports = [validate_file(path, all_paths) for path in outputs]
    failures = 0
    for report in reports:
        if not report.issues:
            continue
        status = "PASS" if report.passed else "FAIL"
        if not report.passed:
            failures += 1
        print(f"\n{status}  {report.path.relative_to(root)}")
        for issue in report.issues:
            print(f"  [{issue.severity.value}] {issue.check}: {issue.message}")

    if failures:
        print(f"\n{failures} active figure artifact(s) failed validation. Deployment blocked.")
        return False

    print(f"\nAll {len(outputs)} active generator artifact(s) passed validation.")
    return True


def cmd_deploy(figure_ids: set[str] | None = None) -> None:
    """Refresh manuscript assets without copying generator intermediates onto the paper surface."""
    from figures.style import load_paths

    root = _repo_root()
    paths_cfg = load_paths()
    paper_dir = root / paths_cfg.get("paper_figures_dir", "paper/figures")
    paper_dir.mkdir(parents=True, exist_ok=True)
    contracts = _load_active_contracts(root, figure_ids=figure_ids)
    outputs = _declared_generator_outputs(root, contracts)

    for f in outputs:
        if not f.exists():
            raise FileNotFoundError(f"Declared generator output missing: {f}")

    removed = 0
    for f in outputs:
        stale_dest = paper_dir / f.name
        if stale_dest.exists():
            stale_dest.unlink()
            removed += 1

    compose_script = root / "scripts" / "paper" / "compose_master_figure3_family.py"
    if not compose_script.exists():
        raise FileNotFoundError(f"Missing manuscript compose script: {compose_script}")

    compose_cmd = [sys.executable, str(compose_script)]
    if figure_ids is not None:
        compose_cmd.extend(["--figures", ",".join(sorted(figure_ids))])
    subprocess.run(
        compose_cmd,
        check=True,
        cwd=str(root),
    )

    print(
        f"Refreshed manuscript assets in {paper_dir}; removed {removed} stale generator artifact(s) from the paper surface."
    )


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
