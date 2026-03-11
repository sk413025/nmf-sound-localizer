# Governance Contracts

This directory defines the operational contracts for the `paper/nature-comm` worktree.

## Layering

- Constitution:
  - `AGENTS.md`
- Canonical task documents:
  - Nature requirements and manuscript workspace docs
- Operational contracts:
  - files in this directory
- Agent operations:
  - `docs/agent-ops/`
- Executable gates:
  - `make paper-check`
  - `scripts/paper/`
- Historical context:
  - `docs/archive/`

## Contracts

- [experiment-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/experiment-contract.md)
- [manuscript-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/manuscript-contract.md)
- [submission-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/submission-contract.md)
- [codex-collaboration-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/codex-collaboration-contract.md)

## When to use what

- Running experiments or preparing results commits:
  - use the experiment contract
- Writing or revising the paper:
  - use the manuscript contract
- Checking Nature-facing compliance or packaging:
  - use the submission contract
- Designing or auditing Codex and multi-agent workflows:
  - use the Codex collaboration contract
- Running multi-agent task orchestration:
  - use `docs/agent-ops/`
