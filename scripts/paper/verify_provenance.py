#!/usr/bin/env python3
"""Verify data-provenance integrity for the Nature Communications paper.

Checks file existence, NPZ/PTH contents, git commit reachability,
symlink integrity, code-state consistency, manuscript residuals,
and cross-references with experiments.yaml.

The machine-readable report is written under ``results/<run_name>/`` by
default so audit artifacts do not pollute the repository root.

Exit code: 0 if no ERROR-level failures, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_LEVELS = {"INFO": 0, "WARN": 1, "ERROR": 2}


class Finding:
    """Single audit finding."""

    def __init__(self, level: str, category: str, message: str) -> None:
        self.level = level
        self.category = category
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "category": self.category, "message": self.message}

    def __str__(self) -> str:
        return f"[{self.level:<5}] {self.category}: {self.message}"


class AuditReport:
    """Accumulates findings and renders them."""

    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add(self, level: str, category: str, message: str) -> None:
        self.findings.append(Finding(level, category, message))

    @property
    def has_errors(self) -> bool:
        return any(f.level == "ERROR" for f in self.findings)

    def summary_counts(self) -> dict[str, int]:
        counts = {"ERROR": 0, "WARN": 0, "INFO": 0}
        for f in self.findings:
            counts[f.level] = counts.get(f.level, 0) + 1
        return counts

    def print_report(self) -> None:
        print("=" * 72)
        print("  Data Provenance Verification Report")
        print("=" * 72)
        for f in self.findings:
            print(f"  {f}")
        print("-" * 72)
        counts = self.summary_counts()
        print(
            f"  Totals: {counts['ERROR']} error(s), "
            f"{counts['WARN']} warning(s), {counts['INFO']} info(s)"
        )
        verdict = "FAIL" if self.has_errors else "PASS"
        print(f"  Verdict: {verdict}")
        print("=" * 72)

    def to_json(self) -> dict[str, Any]:
        return {
            "verdict": "FAIL" if self.has_errors else "PASS",
            "summary": self.summary_counts(),
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Utility: run git commands
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _commit_exists(repo: Path, sha: str) -> bool:
    result = _git(repo, "cat-file", "-t", sha)
    return result.returncode == 0 and result.stdout.strip() == "commit"


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


def check_file_existence(repo: Path, report: AuditReport) -> None:
    """Check 1: required data files exist."""
    required = [
        "h_matrix_normalized_original_to_box.pth",
        "results/figure4_data.json",
        "results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz",
        "results/omp_transformer_speech260_trainval_split_full_20251115_082341/dictionary.npz",
        "results/omp_transformer_speech260_trainval_split_full_20251202_192153/metrics.npz",
        "results/ablate_identity_speech260_seed42_20251210_134919/metrics.npz",
        "paper/figures/fig01_paradigm-shift.jpg",
        "paper/figures/fig04_unrolled-attention-omp.jpg",
        "paper/figures/fig05_structure-macro-selection.png",
        "paper/figures/fig06_angle-specific-mechanism.png",
        "paper/figures/fig07_bandwise-routing-analysis-part1.png",
        "paper/figures/fig08_bandwise-routing-analysis-part2.png",
        "paper/figures/fig09_cross-material-universality.jpg",
    ]
    for rel in required:
        p = repo / rel
        if p.exists():
            report.add("INFO", "file_existence", f"OK  {rel}")
        else:
            report.add("ERROR", "file_existence", f"MISSING  {rel}")


def check_npz_keys(repo: Path, report: AuditReport) -> None:
    """Check 2: NPZ files contain expected keys."""
    try:
        import numpy as np
    except ImportError:
        report.add("ERROR", "npz_keys", "numpy is not installed — cannot verify NPZ files")
        return

    expectations: dict[str, list[str]] = {
        "results/omp_transformer_speech260_trainval_split_full_20251202_192153/metrics.npz": [
            "confusion_matrix",
            "angles",
            "best_accuracy",
        ],
        "results/ablate_identity_speech260_seed42_20251210_134919/metrics.npz": [
            "confusion_matrix",
            "angles",
            "best_accuracy",
        ],
        "results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz": [
            "labels",
            "scores_expert",
            "g_energy_expert",
            "Y_val",
        ],
        "results/omp_transformer_speech260_trainval_split_full_20251115_082341/dictionary.npz": [
            "D",
            "H",
            "angles",
        ],
    }

    for rel, keys in expectations.items():
        p = repo / rel
        if not p.exists():
            report.add("ERROR", "npz_keys", f"Cannot check keys — file missing: {rel}")
            continue
        try:
            data = np.load(str(p), allow_pickle=True)
            present = set(data.files)
            missing = [k for k in keys if k not in present]
            if missing:
                report.add(
                    "ERROR",
                    "npz_keys",
                    f"{rel}: missing keys {missing} (found {sorted(present)})",
                )
            else:
                report.add("INFO", "npz_keys", f"OK  {rel} — required keys present")
        except Exception as exc:
            report.add("ERROR", "npz_keys", f"{rel}: failed to load — {exc}")


def check_pth_loadable(repo: Path, report: AuditReport) -> None:
    """Check 3: PTH file loads and contains expected keys."""
    rel = "h_matrix_normalized_original_to_box.pth"
    p = repo / rel
    if not p.exists():
        report.add("ERROR", "pth_loadable", f"Cannot check — file missing: {rel}")
        return

    try:
        import torch

        data = torch.load(str(p), map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            report.add("ERROR", "pth_loadable", f"{rel}: loaded object is not a dict")
            return
        missing = [k for k in ("H", "angles") if k not in data]
        if missing:
            report.add(
                "ERROR",
                "pth_loadable",
                f"{rel}: missing keys {missing} (found {sorted(data.keys())})",
            )
        else:
            report.add("INFO", "pth_loadable", f"OK  {rel} — contains H, angles")
    except ImportError:
        report.add("WARN", "pth_loadable", "torch is not installed — skipping PTH check")
    except Exception as exc:
        report.add("ERROR", "pth_loadable", f"{rel}: failed to load — {exc}")


def check_git_commits(repo: Path, report: AuditReport) -> None:
    """Check 4: git commit reachability."""
    required = [
        "d9b11d1",
        "c96860b",
        "3785b1f",
        "97942ac",
        "6dd2645",
        "14feb94",
        "b9dcafa",
        "34403c7ea732ef23b77fc38b3afea2427086338e",
    ]
    warn_only = ["15b2981", "88a8940"]

    for sha in required:
        if _commit_exists(repo, sha):
            report.add("INFO", "git_commit", f"OK  {sha}")
        else:
            report.add("ERROR", "git_commit", f"NOT FOUND  {sha}")

    for sha in warn_only:
        if _commit_exists(repo, sha):
            report.add("INFO", "git_commit", f"OK  {sha} (expected-warn)")
        else:
            report.add("WARN", "git_commit", f"NOT FOUND (expected)  {sha}")


def check_symlinks(repo: Path, report: AuditReport) -> None:
    """Check 5: symlink integrity — target exists."""
    symlink_paths = [
        "results/figure4_data.json",
        "results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz",
        "results/omp_transformer_speech260_trainval_split_full_20251115_082341/dictionary.npz",
    ]
    for rel in symlink_paths:
        p = repo / rel
        if not p.is_symlink():
            if p.exists():
                report.add("WARN", "symlink_integrity", f"{rel} exists but is not a symlink")
            else:
                report.add("ERROR", "symlink_integrity", f"MISSING  {rel}")
            continue
        target = p.resolve()
        if target.exists():
            report.add("INFO", "symlink_integrity", f"OK  {rel} → {target}")
        else:
            report.add(
                "ERROR",
                "symlink_integrity",
                f"BROKEN SYMLINK  {rel} → {p.readlink()} (target does not exist)",
            )


def check_code_state(repo: Path, report: AuditReport) -> None:
    """Check 6: code_state.json git_head consistency."""
    checks = [
        (
            "results/omp_transformer_speech260_trainval_split_full_20251115_082341/code_state.json",
            "3785b1fcfed18ad35777b5cc076e4485ab31cd40",
        ),
        (
            "results/ablate_identity_speech260_seed42_20251210_134919/code_state.json",
            "34403c7ea732ef23b77fc38b3afea2427086338e",
        ),
    ]
    for rel, expected_head in checks:
        p = repo / rel
        if not p.exists():
            report.add("WARN", "code_state", f"Cannot check — file missing: {rel}")
            continue
        try:
            data = json.loads(p.read_text())
            actual = data.get("git_head", "")
            if actual.startswith(expected_head) or expected_head.startswith(actual):
                report.add("INFO", "code_state", f"OK  {rel} git_head matches {expected_head[:12]}")
            else:
                report.add(
                    "WARN",
                    "code_state",
                    f"{rel}: git_head mismatch — expected {expected_head[:12]}..., got {actual[:12]}...",
                )
        except Exception as exc:
            report.add("WARN", "code_state", f"{rel}: failed to read — {exc}")


def check_generator_paths(repo: Path, report: AuditReport) -> None:
    """Check 7: generator paths referenced in paths.yaml exist."""
    try:
        import yaml
    except ImportError:
        report.add("WARN", "generator_paths", "PyYAML not installed — skipping paths.yaml check")
        return

    paths_file = repo / "figures" / "conf" / "paths.yaml"
    if not paths_file.exists():
        report.add("WARN", "generator_paths", "figures/conf/paths.yaml not found")
        return

    try:
        data = yaml.safe_load(paths_file.read_text())
    except Exception as exc:
        report.add("WARN", "generator_paths", f"Failed to parse paths.yaml — {exc}")
        return

    def _extract_paths(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
        """Recursively extract string values that look like file paths."""
        results: list[tuple[str, str]] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                results.extend(_extract_paths(v, prefix=f"{prefix}.{k}" if prefix else k))
        elif isinstance(obj, str):
            # Heuristic: treat values containing '/' or ending in known extensions as paths
            if "/" in obj or obj.endswith((".pth", ".npz", ".json", ".yaml", ".py")):
                results.append((prefix, obj))
        return results

    for key, relpath in _extract_paths(data):
        p = repo / relpath
        if p.exists():
            report.add("INFO", "generator_paths", f"OK  paths.yaml[{key}] → {relpath}")
        else:
            report.add("WARN", "generator_paths", f"paths.yaml[{key}] → {relpath} NOT FOUND")


def check_manuscript_tbd(repo: Path, report: AuditReport) -> None:
    """Check 8: count remaining {TBD_...} placeholders in manuscript."""
    manuscript = repo / "paper" / "manuscript" / "manuscript.md"
    if not manuscript.exists():
        report.add("INFO", "manuscript_tbd", "manuscript.md not found — skipping")
        return

    text = manuscript.read_text()
    matches = re.findall(r"\{TBD_[^}]*\}", text)
    count = len(matches)
    if count == 0:
        report.add("INFO", "manuscript_tbd", "No {TBD_...} placeholders remaining")
    else:
        unique = sorted(set(matches))
        report.add(
            "INFO",
            "manuscript_tbd",
            f"{count} {'{TBD_...}'} placeholder(s) remaining: {unique}",
        )


def check_experiments_yaml(repo: Path, report: AuditReport) -> None:
    """Check 9: cross-check experiments.yaml commits with git."""
    try:
        import yaml
    except ImportError:
        report.add("WARN", "experiments_yaml", "PyYAML not installed — skipping")
        return

    exp_file = repo / "figures" / "conf" / "experiments.yaml"
    if not exp_file.exists():
        report.add("WARN", "experiments_yaml", "figures/conf/experiments.yaml not found")
        return

    try:
        data = yaml.safe_load(exp_file.read_text())
    except Exception as exc:
        report.add("WARN", "experiments_yaml", f"Failed to parse experiments.yaml — {exc}")
        return

    # Collect all commit-like strings from the YAML
    sha_pattern = re.compile(r"^[0-9a-f]{7,40}$")

    def _collect_shas(obj: Any) -> list[str]:
        shas: list[str] = []
        if isinstance(obj, dict):
            for v in obj.values():
                shas.extend(_collect_shas(v))
        elif isinstance(obj, list):
            for v in obj:
                shas.extend(_collect_shas(v))
        elif isinstance(obj, str) and sha_pattern.match(obj):
            shas.append(obj)
        return shas

    all_shas = sorted(set(_collect_shas(data)))
    for sha in all_shas:
        if _commit_exists(repo, sha):
            report.add("INFO", "experiments_yaml", f"OK  commit {sha}")
        else:
            report.add("WARN", "experiments_yaml", f"commit {sha} not reachable from this repo")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def detect_repo_root() -> Path:
    """Auto-detect repository root from script location."""
    # scripts/paper/verify_provenance.py  →  go up 3 levels
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir.parent.parent
    if (candidate / ".git").exists():
        return candidate
    # Fallback: ask git
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        cwd=str(script_dir),
    )
    if result.returncode == 0:
        return Path(result.stdout.strip())
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify data-provenance integrity for the Nature Communications paper."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory for audit artifacts "
            "(default: <repo-root>/results/paper_provenance_audit)"
        ),
    )
    args = parser.parse_args()

    repo = (args.repo_root or detect_repo_root()).resolve()
    if not (repo / ".git").exists():
        print(f"ERROR: {repo} does not appear to be a git repository", file=sys.stderr)
        return 1

    print(f"Repo root: {repo}\n")

    report = AuditReport()

    # Run all checks
    check_file_existence(repo, report)
    check_npz_keys(repo, report)
    check_pth_loadable(repo, report)
    check_git_commits(repo, report)
    check_symlinks(repo, report)
    check_code_state(repo, report)
    check_generator_paths(repo, report)
    check_manuscript_tbd(repo, report)
    check_experiments_yaml(repo, report)

    # Print human-readable report
    report.print_report()

    out_dir = args.out_dir or (repo / "results" / "paper_provenance_audit")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON report
    json_path = out_dir / "audit_results.json"
    json_path.write_text(json.dumps(report.to_json(), indent=2) + "\n")
    print(f"\nJSON report written to: {json_path}")

    return 1 if report.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
