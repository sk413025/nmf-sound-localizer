# Governance Contracts

This directory holds the branch's operational contracts.

It is an on-demand layer. Codex should not read this whole directory on first pass. The top-level route is:

1. `AGENTS.md`
2. `.codex/memory/CURRENT_BRANCH_MEMORY.md`
3. `.codex/skills/agent-orchestrator/SKILL.md`

Open a contract here only when the routed task needs it.
Routing still starts at `.codex/skills/agent-orchestrator/SKILL.md`, not inside this directory.

## Canonical contracts

- `experiment-contract.md`
  - experiment execution, reproducibility, and results interpretation
- `manuscript-contract.md`
  - manuscript writing, claim/evidence discipline, and paper-facing explanation
- `scientific-voice-guide.md`
  - canonical positive-style reference for paper-facing prose
- `submission-contract.md`
  - Nature-facing compliance and packaging
- `codex-collaboration-contract.md`
  - anti-duplication, canonical-home policy, and machine-vs-reviewer boundaries
- `closeout-integrity-contract.md`
  - closeout truthfulness, evidence standards, and owner separation
- `runtime-substrate-contract.md`
  - active runtime code surface and substrate limits
- `ASSET_CLASSES.md`
  - asset classification and source-of-truth boundaries

## Use the smallest contract that fits

- writing or revising paper-facing prose:
  - `manuscript-contract.md`
  - `scientific-voice-guide.md`
- checking Nature-facing compliance:
  - `submission-contract.md`
- changing governance, routing, or machine-checkable workflow rules:
  - `codex-collaboration-contract.md`
- reporting completion or evaluating a closeout-sensitive round:
  - `closeout-integrity-contract.md`
- changing active runtime code or runtime-facing scripts:
  - `runtime-substrate-contract.md`

For `high-risk` broader-significance rounds, use:

- `docs/agent-ops/ROUND_GOVERNANCE_SCHEMA.md`
- `results/<round_name>/governance_round.yaml`
- `make paper-governance-gate ROUND_DIR=results/<round_name>`
