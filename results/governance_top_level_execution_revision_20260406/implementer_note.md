# Implementer Note: Top-Level Execution Revision

Date: 2026-04-06

## Files changed

- `AGENTS.md`
- `START_HERE_AGENT.md`
- `docs/governance/codex-collaboration-contract.md`
- `docs/agent-ops/README.md`
- `docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md`
- `docs/agent-ops/TASK_PACKETS.md`
- `docs/agent-ops/ROLE_CATALOG.md`
- `docs/agent-ops/REVIEW_AND_ESCALATION.md`
- `.codex/skills/agent-orchestrator/SKILL.md`
- `scripts/paper/check_governance_links.py`

## Exact old rule summary

The old governance posture was a hard parent-only rule. The key exact statements removed or rewritten were:

- `Treat the top-level agent as the parent orchestrator, not a worker.`
- `The top-level agent is the parent orchestrator and does not execute specialist work directly.`
- `The top-level agent is the parent orchestrator, not a worker.`
- `Even a single bounded task should be handed to at least one child agent.`
- `Act as the top-level parent orchestrator, not a worker.`
- `This parent-only rule applies in both Default mode and Plan mode.`
- `the top-level agent remains the parent orchestrator, not a worker`

## Exact new rule summary

The new governance posture is optional delegation with preserved high-risk separation. The key exact replacement statements now on disk are:

- `Treat the top-level agent as the routing authority: it may execute directly or delegate to child agents after the top-level routing decision.`
- `The top-level agent first classifies the task and decides whether direct execution or delegation is the better fit for scope and risk.`
- `The top-level agent routes through orchestration first, then may execute directly or delegate.`
- `Route the top-level agent through \`agent-orchestrator\` and require an explicit execution-or-delegation decision before specialist work begins.`
- `The top-level agent is the default coordinator for routed work. It may execute directly or delegate to child specialists after an explicit routing decision.`
- `Before issuing any packet, the top-level agent must make an execution-or-delegation decision.`
- `Act as the top-level routing authority. Make an explicit execution-or-delegation decision before specialist work begins.`

The high-risk separation rule was preserved. No change was made to the requirement that `high-risk` rounds keep separate implementer, reviewer, and verifier ownership unless an explicit non-high-risk or compression rationale is recorded.

## Search evidence

Old hard wording search run:

```bash
rg -n "parent orchestrator, not a worker|does not execute specialist work directly|parent-only rule|must stay a parent orchestrator|Even a single bounded task should be handed to at least one child agent|performing specialist execution directly" AGENTS.md START_HERE_AGENT.md docs .codex/skills scripts/paper/check_governance_links.py
```

Result:

- no matches

Final `TASK_PACKETS.md` residue cleanup:

- The paper-asset-review packet now says: `the exact asset-review decisions assigned in this packet, whether to a delegated child or a direct implementer`

Focused `TASK_PACKETS.md` search run:

```bash
rg -n "delegated in this packet" docs/agent-ops/TASK_PACKETS.md
```

Result:

- no matches

Replacement wording search run:

```bash
rg -n "execution-or-delegation|execute directly or delegate|default coordinator for routed work" AGENTS.md START_HERE_AGENT.md docs .codex/skills/agent-orchestrator/SKILL.md scripts/paper/check_governance_links.py
```

Result summary:

- `AGENTS.md` now carries the routing-authority and execution-or-delegation wording.
- `START_HERE_AGENT.md` now tells the top-level agent to decide whether to execute directly or delegate.
- `docs/governance/codex-collaboration-contract.md` now requires an explicit execution-or-delegation decision.
- `docs/agent-ops/README.md`, `docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md`, and `docs/agent-ops/TASK_PACKETS.md` now use the same model.
- `.codex/skills/agent-orchestrator/SKILL.md` now defines an `Execution-or-delegation rule`.
- `scripts/paper/check_governance_links.py` now enforces the new wording instead of the old parent-only wording.

Executable gate check run:

```bash
python scripts/paper/check_governance_links.py
```

Result:

- `OK: governance entrypoints and links are in place.`

## Remaining docs that still imply parent-only behavior

No core docs still imply mandatory child delegation or parent-only execution.

None found in:

- `AGENTS.md`
- `START_HERE_AGENT.md`
- `docs/`
- `.codex/skills/agent-orchestrator/SKILL.md`
- `scripts/paper/check_governance_links.py`

Residual wording note:

- Some docs still use `parent` or `supervisor` as coordination labels for closeout, review selection, and child monitoring.
- Some docs still refer to `child agents`, but those references are conditional on delegation or describe child-role behavior, not a mandatory delegation rule.
- Those remaining references describe coordination ownership, not a ban on direct top-level execution.

## Cleanup addendum

Follow-up consistency cleanup completed for two reviewer-noted residues:

- `START_HERE_AGENT.md` now says: `In Plan mode, both direct and delegated work must stay non-mutating and limited to planning, exploration, checking, or review.`
- `docs/agent-ops/TASK_PACKETS.md` now uses `assigned in this packet, whether to a delegated child or a direct implementer` for the affected `Plan items owned` lines.

Focused residue search run:

```bash
rg -n "delegated child work must stay non-mutating|items delegated in this packet" START_HERE_AGENT.md docs/agent-ops/TASK_PACKETS.md
```

Result:

- no matches
