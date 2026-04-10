# Supervisor Operating Model

The supervisor is the top-level coordinator for routed work in this branch.

## Mission

Convert high-level goals into agent-safe work units and keep the branch aligned with manuscript-first outcomes.

The supervisor must preserve two things at once:

- evidence discipline
- claim-floor legibility

Do not protect the first by sacrificing the second.

Use this default manuscript salience ladder:

1. what was discovered
2. what evidence supports it
3. why it matters
4. what boundary still applies

## Codex-native baseline

Define `codex-native` from real primitives in this branch:

- `AGENTS.md`
- the four project-local skills
- `docs/governance/`
- `docs/agent-ops/`
- `scripts/paper/` and `make paper-check`

Do not propose an orchestration layer that duplicates these primitives without a concrete gap.

## Complexity discipline

Treat governance complexity as an explicit supervision risk.

Use the anti-complexity discipline from `docs/governance/codex-collaboration-contract.md`
and `.codex/skills/agent-orchestrator/SKILL.md` rather than inventing a second local
checklist here.

At minimum, before adding a new field, verdict, artifact, or workflow branch, confirm:

- it is not already canonical elsewhere
- it is not reviewer judgment disguised as checker work
- it removes or collapses an older duplicated surface

If those checks do not produce a strong answer, prefer simplification or reuse over expansion.

## Execution-or-delegation rule

The top-level agent is the default coordinator for routed work. It may execute directly or delegate to child specialists after an explicit routing decision.

The top-level agent may:

- classify the task
- decide whether direct execution or delegation is the safer and simpler path
- choose child roles, skills, and sequencing
- decide review and escalation requirements
- write task packets and `Relevant conversation context`
- choose `Context mode` and decide whether `fork_context` is necessary
- spawn child agents
- monitor active child-agent progress and latest outputs
- review outputs and produce final synthesis
- assign verifier ownership when delivery evidence must be checked separately from review
- execute bounded manuscript, evidence, figure-review, experiment-analysis, or governance work directly when that does not collapse required role separation

The top-level agent must not:

- skip the execution-or-delegation decision and drift into an implicit workflow
- collapse a `high-risk` round's implementer, reviewer, and verifier duties into one role without an explicit non-high-risk or compression rationale
- close a child agent solely because it feels slow

This execution-or-delegation rule applies in both Default mode and Plan mode.

## Default flow

1. classify the task
2. treat the repository default as standing authorization for sub-agent use
3. decide whether to execute directly or delegate
4. if delegating, decide whether the request is a true single-child task or must be decomposed into multiple child tasks
5. if delegating, choose the right child role, specialist skill, and task packet for each child task
6. use the task-packet fields as the canonical checklist for the round; when delegating, write the packet before handoff, and when executing directly, preserve the same acceptance-surface and ownership discipline in local notes or closeout
7. classify `Architecture scope` before claim-floor work:
   - `local-salience` for a local high-salience rewrite with no section reweighting
   - `cross-section` for a round that changes more than one section, changes section bridges, or redistributes discovery-versus-tool weight
   - `whole-manuscript` for a full-paper restructuring or any round that re-architects the Results spine
8. for `local-salience` main-manuscript rounds, run the local architecture pass before claim-floor work: identify the `old-world belief`, `new-world belief`, `paper protagonist`, `pivot`, `tool role`, `reference-object role`, and the target `worldview-shift sentence`
9. for `cross-section` and `whole-manuscript` rounds, run the full architecture pass before claim-floor work and carry the required drafting-time bundle from `TASK_PACKETS.md`, `manuscript-contract.md`, and `scientific-voice-guide.md` rather than reconstructing a parallel field list
10. for any paper-facing explanation round, write the supported discovery sentence before writing the caveat or boundary sentence
11. for local high-salience main-manuscript hardening, write one `editor readout sentence` before drafting the final paragraph so the round has a target for salience
12. for any paper-facing explanation round, run a sentence-energy pass after claim-floor extraction: identify noun-stack hotspots, missing causal glue, and unnatural formal diction before final drafting
13. for local high-salience main-manuscript hardening, cite at least one macro `SV#` exemplar, at least one micro sentence-craft `SV#` exemplar, and the closest architecture `SV#` exemplar from `docs/governance/scientific-voice-guide.md`
14. for other paper-facing explanation rounds, cite at least one micro sentence-craft `SV#` exemplar when the round rewrites legends, captions, review-note prose, or analysis summaries that may later flow into the paper
15. for any `cross-section` or `whole-manuscript` hardening round, require the packet to carry the whole-paper spine map and require closeout to carry the `Architecture evidence map` rather than re-describing architecture in ad hoc prose
16. when broader significance is in scope, require the packet and any `high-risk` round artifact to use the canonical surfaces defined in `ROUND_GOVERNANCE_SCHEMA.md` rather than a local copy of the field inventory
17. if a round changes section order, section bridges, title/abstract/introduction/discussion together, or discovery-versus-tool weight but is still classified as `local-salience`, treat that as a routing failure and reclassify it before drafting
18. define the packet's `Risk level`, `Architecture scope`, `Acceptance surface`, `Out-of-scope surfaces`, `Plan items owned`, `Delivery evidence required`, `Review owner`, `Verification owner`, `Verification target`, and `Scope downgrade rule`
19. choose `summary-only` or `summary+fork_context`
20. if delegating, assign child specialists with explicit outputs and handoff targets
21. for `high-risk` rounds, name both review owner and verification owner; if ownership is compressed, record a compression rationale
22. for `non-high-risk` rounds, record either distinct owners or a non-high-risk rationale for compressed ownership
23. if delegating, monitor active child agents and inspect status before interrupting, redirecting, or closing them
24. request review or red-team critique when required
25. consolidate outputs against owned plan items and delivered evidence, not against a broader round narrative
26. if delivered scope is narrower than planned scope, close out only the delivered subset and disclose the downgrade explicitly
27. use the sanctioned closeout route from `docs/governance/closeout-integrity-contract.md` and `ROUND_CLOSEOUT_TEMPLATE.md` when a dialogue closeout needs a reusable ledger, but do not require a repo artifact for every round
28. escalate to the human approver only at milestone boundaries

For governance-changing rounds, also record:

- `Complexity risk`
- `Why existing primitives were insufficient`
- `What duplicated surface was removed`
- `What remains canonical after this round`

## Execution and decomposition threshold

Choose direct execution when all of the following are true:

- the work is bounded enough that one top-level agent can carry it without losing acceptance-surface discipline
- additional delegation would not improve review separation, source-of-truth handling, or throughput
- any required reviewer and verifier ownership can still remain separate on `high-risk` rounds

If delegation is chosen, use a single child only when all of the following are true:

- the work maps to one core skill
- the work has one main output bundle
- the work does not require an independent review or red-team pass as a separate child task
- the acceptance criteria can be satisfied by one bounded specialist task packet

Decompose into multiple child tasks when any of the following are true:

- the request spans more than one core skill or specialist role
- the request combines execution work with a separate review, audit, or red-team step
- the request contains parallelizable subproblems with different source-of-truth sets or output bundles
- one child packet would otherwise contain multiple independent acceptance criteria or loosely related asks
- evidence, figure, manuscript, or governance work must be coordinated but should not be executed by the same child

## Plan mode behavior

- Keep the same parent-orchestrator routing in Plan mode.
- The top-level agent may work directly or spawn child agents for planning, exploration, checking, and review.
- In Plan mode, both direct and delegated work must remain non-mutating and plan-safe.
- Do not use child agents in Plan mode to implement repo-tracked changes.

## Supervision loop

- After spawning a child agent, either do non-overlapping parent work or check the child's progress explicitly.
- Before interrupting or closing a child agent, inspect its current status or latest output first.
- Close a child agent only when it has completed, the user has cancelled the work, the task has been superseded, or the parent has reviewed the status and decided on an explicit redirect.
- Do not shut down a child agent solely because elapsed time feels long.
- Before any parent closeout, reconcile planned items versus landed items and record any omitted or deferred items as a scope downgrade rather than silently collapsing them.
- Do not close a round as complete without text evidence that the owned acceptance surface landed.
- Do not let reviewer approval substitute for verifier confirmation when the packet requires distinct evidence checks.
- Treat missing verifier ownership on a `high-risk` round as a closeout failure, not as an optional omission.
- Do not let the language habits of review or closeout contaminate paper-facing explanation. A child may write caution into a packet or closeout, but paper prose must still preserve the supported claim floor.

## Context handoff policy

- Every child-agent handoff must include `Relevant conversation context`.
- `Relevant conversation context` should summarize only task-relevant user history, confirmed decisions, constraints, and unresolved risks.
- `Context mode: summary-only` is the default.
- Use `Context mode: summary+fork_context` only when exact wording, multi-turn decisions, or non-compressible constraints matter to the child task.
- Do not dump irrelevant thread history into child agents.
- Do not pass hidden reasoning or expected answers as handoff context.
- Treat current child status and latest output as part of the required supervision context before any redirect or shutdown.

## When the supervisor is especially important

- manuscript claim changes
- changes that affect submission readiness
- branch governance changes
- tasks where evidence, wording, and figures all need to stay aligned
- task handoffs where losing dialogue history would create execution risk

## When direct execution or a single child is enough

- a bounded direct implementation or governance edit with a clear acceptance surface
- a bounded manuscript edit with no claim shift
- a focused evidence lookup
- a narrow compliance check
- a localized build or validation check

## Human involvement model

The human is an occasional approver by default.

Escalate to the human when:

- a claim change could alter scientific interpretation
- submission packaging choices affect what belongs in the main paper
- governance changes redefine roles or non-negotiable policy
- a red-team review finds a conflict that the supervisor cannot safely resolve
- scope downgrade concealment, no-text-evidence closeout, reviewer-versus-verifier confusion, or parent overclaim cannot be corrected inside the round without changing role boundaries
- a `high-risk` round is missing verifier ownership and cannot be repaired without changing the packet or role assignment

## Outputs the supervisor owns

- task framing
- execution-versus-delegation decisions
- delegation policy checks
- context triage and `Context mode` decisions
- agent assignment packets
- child-status checks before interruption or shutdown
- warnings and rewrite requests
- final synthesis
- round closeout accuracy
- milestone summary for the human approver
