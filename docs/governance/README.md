# Governance Contracts

This directory defines the operational contracts for the `paper/nature-comm` worktree.

## Layering

- Constitution:
  - `AGENTS.md`
- Canonical task documents:
  - Nature requirements and manuscript workspace docs
- Operational contracts:
  - files in this directory
- Asset classification:
  - `ASSET_CLASSES.md`
- Agent operations:
- `docs/agent-ops/`
- Executable gates:
  - `make paper-check`
  - `make paper-governance-gate ROUND_DIR=results/<round_name>`
  - `scripts/paper/`
- Historical context:
  - `docs/archive/`

## Contracts

- [experiment-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/experiment-contract.md)
- [manuscript-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/manuscript-contract.md)
- [scientific-voice-guide.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/scientific-voice-guide.md)
- [submission-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/submission-contract.md)
- [codex-collaboration-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/codex-collaboration-contract.md)
- [closeout-integrity-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/closeout-integrity-contract.md)
- [runtime-substrate-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/runtime-substrate-contract.md)
- [ASSET_CLASSES.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/ASSET_CLASSES.md)

## When to use what

- Running experiments or preparing results commits:
  - use the experiment contract
- Writing or revising the paper:
  - use the manuscript contract
  - use the scientific voice guide as the canonical positive-style reference
  - use the scientific voice guide for paper-level architecture, paper-level salience, and sentence-level craft across all paper-facing explanation surfaces
  - treat the manuscript contract as the stricter main-manuscript subset inside the broader paper-facing explanation model
  - preserve the supported claim floor; do not let governance or closeout caution become manuscript tone
- Checking Nature-facing compliance or packaging:
  - use the submission contract
- Designing or auditing Codex and multi-agent workflows:
  - use the Codex collaboration contract
- Reporting completion, review outcome, or milestone status:
  - route through the closeout integrity contract
  - use it as the governing contract for exact-text evidence, scope-downgrade disclosure, and completion-status reporting
- Governing a high-risk round that could change claims, governance posture, or acceptance status:
  - route through the closeout integrity contract
  - classify the round there and use its owner-separation or compression-rationale rules
- Changing active runtime code, scripts, tests, or package metadata:
  - use the runtime substrate contract
  - treat only TF + USM + soft-OMP support paths as active by default
- Running multi-agent task orchestration:
  - use `docs/agent-ops/`
- Governing a `high-risk` broader-significance round:
  - use `docs/agent-ops/ROUND_GOVERNANCE_SCHEMA.md`
  - create `results/<round_name>/governance_round.yaml`
  - treat `make paper-governance-gate ROUND_DIR=results/<round_name>` as blocking
  - treat the schema doc as the only machine-readable field inventory; other contracts should reference it rather than recreate it
