# Reviewer Memo - Governance Top-Level Execution Revision

Date: 2026-04-06

Risk level: high-risk

Review verdict: `pass`

## Findings first

1. The previously flagged wording residues are now resolved. I no longer find a remaining contradiction in the reviewed governance surface that blocks the new execution-or-delegation rule.
   The final `TASK_PACKETS.md` residue at line 160 is now cleared as well.

2. The hard parent-only execution ban is materially removed from the branch entrypoints. The patched governance now explicitly allows the top-level agent to execute directly or delegate after a routing decision.

3. High-risk truth-maintenance remains preserved. I still find no weakening of manuscript-first governance, reviewer / verifier separation, or independent verification requirements.

4. Repo-wide blocker search is now clean for the reviewed phrases. I found no surviving live text in the reviewed governance surface that still says the top-level agent is not a worker, must not execute directly, or must always use at least one child agent.
   The governance link check also passes on the patched tree.

## Delta on previously flagged items

### 1. START_HERE_AGENT.md Plan-mode wording

Prior rule:
- The file previously still framed Plan mode only in terms of delegated child work.

Current patched state:
- It now says: `In Plan mode, both direct and delegated work must stay non-mutating and limited to planning, exploration, checking, or review.`

Status:
- `resolved`

Exact remaining contradiction or gap:
- None.

Evidence:
- [START_HERE_AGENT.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_AGENT.md#L45)

### 2. TASK_PACKETS governance-packet wording

Prior rule:
- The governance/orchestration packet previously still used `delegated in this packet` wording.

Current patched state:
- It now says:
  - `the exact orchestration, review, or hardening items assigned in this packet, whether to a delegated child or a direct implementer`

Status:
- `resolved`

Exact remaining contradiction or gap:
- None.

Evidence:
- [TASK_PACKETS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/TASK_PACKETS.md#L208)
- [TASK_PACKETS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/TASK_PACKETS.md#L209)

### 3. TASK_PACKETS paper-asset-review wording

Prior rule:
- The paper-asset-review packet previously still said `the exact asset-review decisions delegated in this packet`.

Current patched state:
- It now says:
  - `the exact asset-review decisions assigned in this packet, whether to a delegated child or a direct implementer`

Status:
- `resolved`

Exact remaining contradiction or gap:
- None.

Evidence:
- [TASK_PACKETS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/TASK_PACKETS.md#L159)
- [TASK_PACKETS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/TASK_PACKETS.md#L160)

## Surface summary

### 1. AGENTS.md

Prior rule:
- The top-level agent was the parent orchestrator, not a worker, and did not execute specialist work directly.

Current patched state:
- The file now explicitly says the top-level agent is the routing authority and `may execute directly or delegate to child agents after the top-level routing decision`.

Status:
- `resolved`

Exact remaining contradiction or gap:
- None found.

Evidence:
- [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md#L58)
- [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md#L59)
- [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md#L76)

### 2. START_HERE_AGENT.md

Prior rule:
- The top-level agent was not a worker and even a single bounded task should be handed to at least one child agent.

Current patched state:
- The file now says the top-level agent routes through orchestration first, then may execute directly or delegate, and that delegation is conditional rather than mandatory.

Status:
- `resolved`

Exact remaining contradiction or gap:
- None found.

Evidence:
- [START_HERE_AGENT.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_AGENT.md#L21)
- [START_HERE_AGENT.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_AGENT.md#L33)
- [START_HERE_AGENT.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_AGENT.md#L34)

### 3. Codex Collaboration Contract

Prior rule:
- The top-level agent was treated as a parent orchestrator, not a worker, with decomposition framed only in child-task terms.

Current patched state:
- The contract now requires an explicit `execution-or-delegation decision before specialist work begins`, and decomposition is conditional on delegation being chosen.

Status:
- `resolved`

Exact remaining contradiction or gap:
- None found.

Evidence:
- [docs/governance/codex-collaboration-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/codex-collaboration-contract.md#L21)
- [docs/governance/codex-collaboration-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/codex-collaboration-contract.md#L23)

### 4. Supervisor Operating Model

Prior rule:
- Explicit `Parent-only rule`; the top-level agent was not a worker and must not perform specialist execution.

Current patched state:
- The rule is now `Execution-or-delegation rule`.
- The file explicitly says the top-level agent may execute directly or delegate after an explicit routing decision.

Status:
- `resolved`

Exact remaining contradiction or gap:
- None found.

Evidence:
- [docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md#L21)
- [docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md#L23)
- [docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md#L37)

### 5. agent-orchestrator skill

Prior rule:
- Act as top-level parent orchestrator, not a worker, and do not perform specialist execution yourself.

Current patched state:
- The skill now says to act as the top-level routing authority and make an explicit execution-or-delegation decision before specialist work begins.
- It explicitly allows direct execution when bounded and when review / verification separation is preserved.

Status:
- `resolved`

Exact remaining contradiction or gap:
- None found.

Evidence:
- [.codex/skills/agent-orchestrator/SKILL.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/skills/agent-orchestrator/SKILL.md#L30)
- [.codex/skills/agent-orchestrator/SKILL.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/skills/agent-orchestrator/SKILL.md#L32)
- [.codex/skills/agent-orchestrator/SKILL.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/skills/agent-orchestrator/SKILL.md#L45)

### 6. TASK_PACKETS.md

Prior rule:
- Packet system assumed child delegation by default and encoded several child-only phrases.

Current patched state:
- The file now supports both delegated tasks and direct top-level execution rounds.
- The packet language reviewed here now uses `assigned ... whether to a delegated child or a direct implementer`.

Status:
- `resolved`

Exact remaining contradiction or gap:
- None found in the reviewed packet surfaces.

Evidence:
- [TASK_PACKETS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/TASK_PACKETS.md#L3)
- [TASK_PACKETS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/TASK_PACKETS.md#L29)
- [TASK_PACKETS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/TASK_PACKETS.md#L66)
- [TASK_PACKETS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/TASK_PACKETS.md#L160)
- [TASK_PACKETS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/TASK_PACKETS.md#L209)

### 7. High-risk separation and manuscript-first governance

Prior rule:
- High-risk rounds must preserve implementer / reviewer / verifier separation unless justified otherwise.
- Reviewer pass must not substitute for verification.
- Manuscript-first governance must remain intact.

Current patched state:
- These protections remain intact in the patched governance surface.

Status:
- `resolved`

Exact remaining contradiction or gap:
- None found.

Evidence:
- [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md#L92)
- [docs/governance/closeout-integrity-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/closeout-integrity-contract.md)
- [docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md#L58)
- [.codex/skills/agent-orchestrator/SKILL.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/skills/agent-orchestrator/SKILL.md#L74)

## Does the patch really give the user what they asked for?

Yes.

The current governance explicitly permits direct top-level execution after a routing decision. The old hard parent-only prohibition is gone from the constitution, quickstart, collaboration contract, supervisor model, orchestrator skill, related operations docs, and governance-link check.

## Remaining contradiction that blocks the new rule

None found.

I do not find a remaining contradiction in the reviewed governance surface that still blocks the new execution-or-delegation rule.

## Final judgment

The governance revision now lands cleanly on the reviewed acceptance surface:
- hard parent-only rule removed or rewritten everywhere material that I reviewed
- new rule clearly states that the top-level agent may execute directly or delegate
- no contradiction remains across the reviewed core entrypoints
- high-risk reviewer / verifier separation remains preserved
- manuscript-first governance remains preserved
