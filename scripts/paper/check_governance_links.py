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
    (REPO_ROOT / "README.md", ".codex/skills/agent-orchestrator/SKILL.md"),
    (REPO_ROOT / "AGENTS.md", "docs/governance/README.md"),
    (REPO_ROOT / "AGENTS.md", "Governance Precedence"),
    (REPO_ROOT / "AGENTS.md", "docs/agent-ops/README.md"),
    (REPO_ROOT / "AGENTS.md", "runtime-substrate-contract.md"),
    (REPO_ROOT / "AGENTS.md", "Route top-level work through `agent-orchestrator` first."),
    (REPO_ROOT / "AGENTS.md", "The top-level agent is the parent orchestrator"),
    (REPO_ROOT / "AGENTS.md", "standing authorization for sub-agent use"),
    (REPO_ROOT / "AGENTS.md", "Apply this parent-orchestrator policy in both Default mode and Plan mode."),
    (REPO_ROOT / "AGENTS.md", "The parent must inspect a child agent's current status or latest output before interrupting or closing it."),
    (REPO_ROOT / "AGENTS.md", "The parent must decide whether a request is a true single-child task or must be decomposed into multiple child tasks."),
    (REPO_ROOT / "CONTRIBUTING.md", "docs/governance/README.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", "docs/agent-ops/README.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", ".codex/skills/paper-submission/SKILL.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", ".codex/skills/agent-orchestrator/SKILL.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", "docs/agent-ops/TASK_PACKETS.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", "Context mode: `summary-only`"),
    (REPO_ROOT / "START_HERE_AGENT.md", "standing authorization for sub-agent use"),
    (REPO_ROOT / "START_HERE_AGENT.md", "Apply this parent-orchestrator policy in both Default mode and Plan mode."),
    (REPO_ROOT / "START_HERE_AGENT.md", "Inspect a child agent's current status or latest output before interrupting or closing it."),
    (REPO_ROOT / "START_HERE_AGENT.md", "Use a single child only when the request is genuinely single-scope; otherwise split it into multiple child tasks before execution starts."),
    (REPO_ROOT / "docs/governance/codex-collaboration-contract.md", "Relevant conversation context"),
    (REPO_ROOT / "docs/governance/codex-collaboration-contract.md", "`summary+fork_context`"),
    (REPO_ROOT / "docs/governance/codex-collaboration-contract.md", "standing authorization for sub-agent use"),
    (REPO_ROOT / "docs/governance/codex-collaboration-contract.md", "monitor active child agents and inspect status before interrupting or closing them"),
    (REPO_ROOT / "docs/governance/codex-collaboration-contract.md", "The top-level parent must make an explicit decomposition decision before execution"),
    (REPO_ROOT / "docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md", "The top-level agent is the parent orchestrator, not a worker."),
    (REPO_ROOT / "docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md", "standing authorization for sub-agent use"),
    (REPO_ROOT / "docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md", "This parent-only rule applies in both Default mode and Plan mode."),
    (REPO_ROOT / "docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md", "In Plan mode, both parent and child work must remain non-mutating and plan-safe."),
    (REPO_ROOT / "docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md", "Before interrupting or closing a child agent, inspect its current status or latest output first."),
    (REPO_ROOT / "docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md", "## Decomposition threshold"),
    (REPO_ROOT / "docs/agent-ops/TASK_PACKETS.md", "Relevant conversation context"),
    (REPO_ROOT / "docs/agent-ops/TASK_PACKETS.md", "Context mode"),
    (REPO_ROOT / "docs/agent-ops/TASK_PACKETS.md", "standing authorization for sub-agent use"),
    (REPO_ROOT / "docs/agent-ops/TASK_PACKETS.md", "Before issuing any packet, the parent must make a decomposition decision."),
    (REPO_ROOT / ".codex" / "skills" / "agent-orchestrator" / "SKILL.md", "Act as the top-level parent orchestrator, not a worker."),
    (REPO_ROOT / ".codex" / "skills" / "agent-orchestrator" / "SKILL.md", "standing authorization for sub-agent use"),
    (REPO_ROOT / ".codex" / "skills" / "agent-orchestrator" / "SKILL.md", "Apply this parent-only rule in both Default mode and Plan mode."),
    (REPO_ROOT / ".codex" / "skills" / "agent-orchestrator" / "SKILL.md", "Keep parent and child work non-mutating and plan-safe until execution mode."),
    (REPO_ROOT / ".codex" / "skills" / "agent-orchestrator" / "SKILL.md", "Before interrupting or closing a child agent, inspect its current status or latest output first."),
    (REPO_ROOT / ".codex" / "skills" / "agent-orchestrator" / "SKILL.md", "## Decomposition decision"),
    (REPO_ROOT / "README.md", "ASSET_CLASSES.md"),
]

STALE_STRING_CHECKS = [
    (REPO_ROOT / "README.md", "Nature Figure Workspace"),
    (REPO_ROOT / "README.md", "docs/codex-native-assessment/README.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", "Choose one skill only:"),
    (REPO_ROOT / "AGENTS.md", "stop at the orchestration gate"),
    (REPO_ROOT / "START_HERE_AGENT.md", "stop at the orchestration gate"),
    (REPO_ROOT / "docs/governance/codex-collaboration-contract.md", "stop at an orchestration gate"),
    (REPO_ROOT / "docs/agent-ops/TASK_PACKETS.md", "stop at the orchestration gate"),
    (REPO_ROOT / ".codex" / "skills" / "agent-orchestrator" / "SKILL.md", "stop at the orchestration gate"),
    (REPO_ROOT / "START_HERE_AGENT.md", ".codex/skills/manuscript-revision/SKILL.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", ".codex/skills/claim-evidence-audit/SKILL.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", ".codex/skills/results-interpretation/SKILL.md"),
    (REPO_ROOT / "START_HERE_AGENT.md", ".codex/skills/codex-native-assessment/SKILL.md"),
    (REPO_ROOT / "START_HERE_HUMAN.md", "docs/codex-native-assessment/README.md"),
    (REPO_ROOT / "docs" / "agent-ops" / "SUPERVISOR_OPERATING_MODEL.md", "When a single agent is enough"),
    (REPO_ROOT / "docs" / "agent-ops" / "SUPERVISOR_OPERATING_MODEL.md", "decide whether the task is single-agent or multi-agent"),
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
