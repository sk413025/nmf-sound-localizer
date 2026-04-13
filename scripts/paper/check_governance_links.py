#!/usr/bin/env python3
"""Check that the branch-level governance skeleton is present and canonically linked."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    REPO_ROOT / "START_HERE_AGENT.md",
    REPO_ROOT / "START_HERE_HUMAN.md",
    REPO_ROOT / ".codex" / "memory" / "README.md",
    REPO_ROOT / ".codex" / "memory" / "CURRENT_BRANCH_MEMORY.md",
    REPO_ROOT / "docs" / "governance" / "README.md",
    REPO_ROOT / "docs" / "governance" / "experiment-contract.md",
    REPO_ROOT / "docs" / "governance" / "manuscript-contract.md",
    REPO_ROOT / "docs" / "governance" / "scientific-voice-guide.md",
    REPO_ROOT / "docs" / "governance" / "submission-contract.md",
    REPO_ROOT / "docs" / "governance" / "codex-collaboration-contract.md",
    REPO_ROOT / "docs" / "governance" / "closeout-integrity-contract.md",
    REPO_ROOT / "docs" / "governance" / "runtime-substrate-contract.md",
    REPO_ROOT / "docs" / "governance" / "ASSET_CLASSES.md",
    REPO_ROOT / "docs" / "agent-ops" / "README.md",
    REPO_ROOT / "docs" / "agent-ops" / "SUPERVISOR_OPERATING_MODEL.md",
    REPO_ROOT / "docs" / "agent-ops" / "ROLE_CATALOG.md",
    REPO_ROOT / "docs" / "agent-ops" / "ROUND_CLOSEOUT_TEMPLATE.md",
    REPO_ROOT / "docs" / "agent-ops" / "ROUND_GOVERNANCE_SCHEMA.md",
    REPO_ROOT / "docs" / "agent-ops" / "TASK_PACKETS.md",
    REPO_ROOT / "docs" / "agent-ops" / "REVIEW_AND_ESCALATION.md",
    REPO_ROOT / "docs" / "evidence" / "README.md",
    REPO_ROOT / "docs" / "working-notes" / "README.md",
    REPO_ROOT / "legacy" / "README.md",
    REPO_ROOT / "scripts" / "paper" / "check_asset_boundaries.py",
    REPO_ROOT / "scripts" / "paper" / "check_figure_references.py",
    REPO_ROOT / "scripts" / "paper" / "check_round_governance_semantics.py",
    REPO_ROOT / "scripts" / "paper" / "rerun_active_figures.py",
    REPO_ROOT / "scripts" / "paper" / "verify_provenance.py",
    REPO_ROOT / "DATA_PROVENANCE.md",
    REPO_ROOT / "paper" / "figures" / "README.md",
    REPO_ROOT / "paper" / "manuscript" / "FIGURE_NAMING_CONTRACT.md",
    REPO_ROOT / "figures" / "FIGURE_REGISTRY.md",
    REPO_ROOT / "figures" / "conf" / "experiments.yaml",
    REPO_ROOT / "figures" / "conf" / "review_targets.yaml",
    REPO_ROOT / "figures" / "conf" / "fig04_fig06_panel_name_provenance.md",
    REPO_ROOT / "figures" / "fig04_stepwise_mechanics.py",
    REPO_ROOT / "figures" / "generators" / "fig04_solver_dynamics.py",
    REPO_ROOT / ".codex" / "skills" / "paper-submission" / "SKILL.md",
    REPO_ROOT / ".codex" / "skills" / "paper-asset-review" / "SKILL.md",
    REPO_ROOT / ".codex" / "skills" / "experiment-results" / "SKILL.md",
    REPO_ROOT / ".codex" / "skills" / "agent-orchestrator" / "SKILL.md",
]

# Keep the link gate structural: each doc needs only enough anchors to prove
# it points at the right canonical surface and still carries its role-specific job.
ANCHOR_CHECKS = {
    REPO_ROOT / "README.md": [
        "docs/governance/README.md",
        "docs/agent-ops/README.md",
        ".codex/skills/agent-orchestrator/SKILL.md",
    ],
    REPO_ROOT / "AGENTS.md": [
        "Governance Precedence",
        ".codex/memory/CURRENT_BRANCH_MEMORY.md",
        "Route top-level work through `agent-orchestrator` first.",
        "make paper-governance-gate ROUND_DIR=results/<round_name>",
        "Treat governance complexity as a branch-level failure mode.",
    ],
    REPO_ROOT / "START_HERE_AGENT.md": [
        ".codex/memory/CURRENT_BRANCH_MEMORY.md",
        "docs/agent-ops/README.md",
        ".codex/skills/agent-orchestrator/SKILL.md",
        "Context mode: `summary-only`",
        "governance_round.yaml",
    ],
    REPO_ROOT / ".codex" / "memory" / "README.md": [
        "not a second constitution",
        "CURRENT_BRANCH_MEMORY.md",
        "must not create new policy",
    ],
    REPO_ROOT / ".codex" / "memory" / "CURRENT_BRANCH_MEMORY.md": [
        "Current Nature Communications Framing",
        "second-layer discovery",
        "Read-Next Links",
    ],
    REPO_ROOT / "START_HERE_HUMAN.md": [
        "docs/governance/scientific-voice-guide.md",
        "noun stacking",
        "paper-level architecture",
    ],
    REPO_ROOT / "docs" / "governance" / "README.md": [
        "scientific-voice-guide.md",
        "codex-collaboration-contract.md",
        "closeout-integrity-contract.md",
    ],
    REPO_ROOT / "docs" / "governance" / "scientific-voice-guide.md": [
        "one endogenous second-layer discovery",
        "ROUND_GOVERNANCE_SCHEMA.md",
        "Sentence energy is not enough.",
    ],
    REPO_ROOT / "docs" / "governance" / "manuscript-contract.md": [
        "governance_round.yaml",
        "ROUND_GOVERNANCE_SCHEMA.md",
        "Do not recreate a second schema",
    ],
    REPO_ROOT / "docs" / "governance" / "codex-collaboration-contract.md": [
        "one canonical machine-readable artifact at `results/<round_name>/governance_round.yaml`",
        "Do not let one concept acquire multiple canonical homes.",
        "Repo-local branch memory may summarize stable branch state",
        "Do not let the checker over-model reviewer judgment.",
    ],
    REPO_ROOT / "docs" / "governance" / "closeout-integrity-contract.md": [
        "governance_round.yaml",
        "independent verification",
        "exact text or exact diff evidence",
    ],
    REPO_ROOT / "docs" / "agent-ops" / "README.md": [
        ".codex/memory/CURRENT_BRANCH_MEMORY.md",
        "NATURE_REVIEWER_STACK.md",
        "ROUND_GOVERNANCE_SCHEMA.md",
        "governance_round.yaml",
    ],
    REPO_ROOT / "docs" / "agent-ops" / "SUPERVISOR_OPERATING_MODEL.md": [
        "Complexity discipline",
        "ROUND_GOVERNANCE_SCHEMA.md",
        "execution-or-delegation rule",
    ],
    REPO_ROOT / "docs" / "agent-ops" / "TASK_PACKETS.md": [
        "Use packet fields to record decisions, not to recreate a second schema.",
        "ROUND_GOVERNANCE_SCHEMA.md",
        "Before issuing any packet, the top-level agent must make an execution-or-delegation decision.",
    ],
    REPO_ROOT / "docs" / "agent-ops" / "ROUND_GOVERNANCE_SCHEMA.md": [
        "The semantic gate checks only:",
        "does **not** try to infer reviewer judgment",
        "make paper-governance-gate",
    ],
    REPO_ROOT / "docs" / "agent-ops" / "NATURE_REVIEWER_STACK.md": [
        "Reviewed status",
        "only machine-relevant level decision",
        "not as a second status ladder",
    ],
    REPO_ROOT / "docs" / "agent-ops" / "ROUND_CLOSEOUT_TEMPLATE.md": [
        "blocking repo artifact is `results/<round_name>/governance_round.yaml`",
        "human-facing closeout verdict, not a canonical YAML field",
        "Do not let closeout freehand a second governance vocabulary.",
    ],
    REPO_ROOT / "docs" / "agent-ops" / "REVIEW_AND_ESCALATION.md": [
        "WARN_OVERMODELED_GOVERNANCE",
        "WARN_PROTAGONIST_DRIFT",
        "SV#",
    ],
    REPO_ROOT / ".codex" / "skills" / "agent-orchestrator" / "SKILL.md": [
        ".codex/memory/CURRENT_BRANCH_MEMORY.md",
        "Complexity self-check",
        "ROUND_GOVERNANCE_SCHEMA.md",
        "governance_round.yaml",
    ],
    REPO_ROOT / ".codex" / "skills" / "paper-submission" / "SKILL.md": [
        "docs/governance/scientific-voice-guide.md",
        "full architecture pass",
        "ROUND_GOVERNANCE_SCHEMA.md",
    ],
    REPO_ROOT / "paper" / "figures" / "README.md": [
        "paper-facing asset surface",
    ],
    REPO_ROOT / "scripts" / "paper" / "check_figure_references.py": [
        "paper-facing figure references",
    ],
    REPO_ROOT / "scripts" / "paper" / "verify_provenance.py": [
        "paper-facing asset list",
    ],
}

STALE_ANCHORS = {
    REPO_ROOT / "README.md": [
        "Nature Figure Workspace",
        "docs/codex-native-assessment/README.md",
    ],
    REPO_ROOT / "AGENTS.md": [
        "stop at the orchestration gate",
    ],
    REPO_ROOT / "START_HERE_AGENT.md": [
        "Choose one skill only:",
        "stop at the orchestration gate",
        ".codex/skills/manuscript-revision/SKILL.md",
        ".codex/skills/claim-evidence-audit/SKILL.md",
        ".codex/skills/results-interpretation/SKILL.md",
        ".codex/skills/codex-native-assessment/SKILL.md",
    ],
    REPO_ROOT / "START_HERE_HUMAN.md": [
        "docs/codex-native-assessment/README.md",
    ],
    REPO_ROOT / "docs" / "agent-ops" / "SUPERVISOR_OPERATING_MODEL.md": [
        "When a single agent is enough",
        "decide whether the task is single-agent or multi-agent",
    ],
    REPO_ROOT / "docs" / "agent-ops" / "TASK_PACKETS.md": [
        "Discussion worldview-shift sentence",
        "stop at the orchestration gate",
    ],
    REPO_ROOT / "docs" / "governance" / "scientific-voice-guide.md": [
        "manuscript-facing prose",
    ],
}


def main() -> int:
    errors: list[str] = []

    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"missing required governance file: {path.relative_to(REPO_ROOT)}")

    for path, needles in ANCHOR_CHECKS.items():
        if not path.exists():
            errors.append(f"missing file for anchor check: {path.relative_to(REPO_ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                errors.append(f"{path.relative_to(REPO_ROOT)} is missing canonical anchor: {needle}")

    for path, needles in STALE_ANCHORS.items():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle in text:
                errors.append(f"{path.relative_to(REPO_ROOT)} still contains stale identity text: {needle}")

    if errors:
        print("ERROR: governance link check failed.")
        for error in errors:
            print(f"- {error}")
        return 1

    print("OK: governance entrypoints and canonical anchors are in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
