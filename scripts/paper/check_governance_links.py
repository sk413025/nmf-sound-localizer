#!/usr/bin/env python3
"""Check that the branch-level governance skeleton is present and linked."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    REPO_ROOT / "START_HERE_AGENT.md",
    REPO_ROOT / "START_HERE_HUMAN.md",
    REPO_ROOT / "docs" / "governance" / "README.md",
    REPO_ROOT / "docs" / "governance" / "experiment-contract.md",
    REPO_ROOT / "docs" / "governance" / "manuscript-contract.md",
    REPO_ROOT / "docs" / "governance" / "submission-contract.md",
    REPO_ROOT / "docs" / "governance" / "codex-collaboration-contract.md",
    REPO_ROOT / "docs" / "governance" / "runtime-substrate-contract.md",
    REPO_ROOT / "docs" / "governance" / "ASSET_CLASSES.md",
    REPO_ROOT / "docs" / "agent-ops" / "README.md",
    REPO_ROOT / "docs" / "agent-ops" / "SUPERVISOR_OPERATING_MODEL.md",
    REPO_ROOT / "docs" / "agent-ops" / "ROLE_CATALOG.md",
    REPO_ROOT / "docs" / "agent-ops" / "TASK_PACKETS.md",
    REPO_ROOT / "docs" / "agent-ops" / "REVIEW_AND_ESCALATION.md",
    REPO_ROOT / "docs" / "evidence" / "README.md",
    REPO_ROOT / "docs" / "working-notes" / "README.md",
    REPO_ROOT / "legacy" / "README.md",
    REPO_ROOT / "scripts" / "paper" / "check_asset_boundaries.py",
    REPO_ROOT / ".codex" / "skills" / "paper-submission" / "SKILL.md",
    REPO_ROOT / ".codex" / "skills" / "paper-asset-review" / "SKILL.md",
    REPO_ROOT / ".codex" / "skills" / "experiment-results" / "SKILL.md",
    REPO_ROOT / ".codex" / "skills" / "agent-orchestrator" / "SKILL.md",
]

STRING_CHECKS = [
    (REPO_ROOT / "README.md", "START_HERE_AGENT.md"),
    (REPO_ROOT / "README.md", "docs/governance/README.md"),
    (REPO_ROOT / "README.md", "docs/agent-ops/README.md"),
    (REPO_ROOT / "AGENTS.md", "docs/governance/README.md"),
    (REPO_ROOT / "AGENTS.md", "Governance Precedence"),
    (REPO_ROOT / "AGENTS.md", "docs/agent-ops/README.md"),
    (REPO_ROOT / "AGENTS.md", "runtime-substrate-contract.md"),
    (REPO_ROOT / "CONTRIBUTING.md", "docs/governance/README.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", "docs/agent-ops/README.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", ".codex/skills/paper-submission/SKILL.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", "docs/agent-ops/TASK_PACKETS.md"),
    (REPO_ROOT / "README.md", "ASSET_CLASSES.md"),
]

STALE_STRING_CHECKS = [
    (REPO_ROOT / "README.md", "Nature Figure Workspace"),
    (REPO_ROOT / "README.md", "docs/codex-native-assessment/README.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", ".codex/skills/manuscript-revision/SKILL.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", ".codex/skills/claim-evidence-audit/SKILL.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", ".codex/skills/results-interpretation/SKILL.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", ".codex/skills/codex-native-assessment/SKILL.md"),
    (REPO_ROOT / "START_HERE_HUMAN.md", "docs/codex-native-assessment/README.md"),
]


def main() -> int:
    errors: list[str] = []

    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"missing required governance file: {path.relative_to(REPO_ROOT)}")

    for path, needle in STRING_CHECKS:
        if not path.exists():
            errors.append(f"missing file for string check: {path.relative_to(REPO_ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        if needle not in text:
            errors.append(f"{path.relative_to(REPO_ROOT)} is missing required reference: {needle}")

    for path, needle in STALE_STRING_CHECKS:
        if path.exists() and needle in path.read_text(encoding="utf-8"):
            errors.append(f"{path.relative_to(REPO_ROOT)} still contains stale identity text: {needle}")

    if errors:
        print("ERROR: governance link check failed.")
        for error in errors:
            print(f"- {error}")
        return 1

    print("OK: governance entrypoints and links are in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
