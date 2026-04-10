---
name: agent-orchestrator
description: Top-level orchestration for this repository. Use when Codex is the first agent receiving a task in this branch, or when coordinating child agents, defining role boundaries, choosing task packets, managing review loops, or routing manuscript-first requests. The top-level agent must route through orchestration first, then decide whether direct execution or delegation is the better fit.
---

# Agent Orchestrator

Use this skill for:

- top-level parent orchestration
- supervisor-led task decomposition
- child-agent coordination
- review and escalation planning
- routing tasks to roles and skills

## Required first step

Open and follow:

- `START_HERE_AGENT.md`
- `docs/governance/codex-collaboration-contract.md`
- `docs/governance/closeout-integrity-contract.md`
- `docs/governance/scientific-voice-guide.md`
- `docs/agent-ops/README.md`
- `docs/agent-ops/NATURE_REVIEWER_STACK.md`
- `docs/agent-ops/ROUND_CLOSEOUT_TEMPLATE.md`
- `docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md`
- `docs/agent-ops/TASK_PACKETS.md`
- `docs/agent-ops/REVIEW_AND_ESCALATION.md`

## Execution-or-delegation rule

Act as the top-level routing authority. Make an explicit execution-or-delegation decision before specialist work begins.

Do:

- classify the task
- decide whether direct execution or delegation best fits scope, risk, and throughput
- if delegating, choose the child role, skill, and task packet
- if delegating, write `Relevant conversation context`
- choose `Context mode`
- decide whether `fork_context` is necessary
- if delegating, spawn child agents
- if delegating, monitor active child-agent progress and latest outputs
- review outputs and synthesize results
- execute directly when the task is bounded and direct execution does not collapse required review or verification separation

Do not:

- skip the execution-or-delegation decision and drift into an implicit workflow
- collapse `high-risk` implementer, reviewer, and verifier ownership into one role without an explicit non-high-risk or compression rationale
- dump irrelevant thread history into a child prompt
- close a child agent solely because it feels slow

Apply this execution-or-delegation rule in both Default mode and Plan mode.

## Workflow

1. Classify the task by manuscript impact and role complexity.
2. Treat this repository's default operating mode as standing authorization for sub-agent use.
3. Decide whether direct execution or delegation is the better fit.
4. If delegating, decide whether the request is a valid single-child task or must be decomposed into multiple child tasks.
5. If delegating, choose the right child task packet, role, and core skill for each child task.
6. Use the task-packet fields as the canonical checklist for the round; when delegating, write the packet before handoff, and when executing directly, preserve the same acceptance-surface and ownership discipline in local notes or closeout.
7. For paper-facing explanation tasks, state the prose acceptance rule explicitly in the packet or review request: prefer scientific-inference prose that moves by `observation -> inference -> bounded conclusion`, with active voice, simple cause-effect relations, lower noun-stack friction, natural enough scientific English for one-pass reading, and narrative architecture that delivers one cognitive shift rather than an experiment log.
8. Classify `Architecture scope` before claim-floor work:
   - `local-salience` for a local high-salience rewrite with no section reweighting
   - `cross-section` for a round that changes more than one section, section bridges, or discovery-versus-tool weight
   - `whole-manuscript` for a full-paper restructuring or any round that re-architects the Results spine
9. For `local-salience` main-manuscript hardening, run the local architecture pass: identify the `old-world belief`, `new-world belief`, `paper protagonist`, `pivot`, `tool role`, `reference-object role`, and target `worldview-shift sentence`.
10. For `cross-section` and `whole-manuscript` hardening, run the full architecture pass: identify the `old-world belief`, `new-world belief`, `paper protagonist`, `supporting actors`, `paper spine map`, `Results section jobs`, `pivot`, `pivot sentence`, `discovery cash-out section`, `tool role`, `reference-object role`, `discovery-vs-tool weight budget`, `redundancy / breathing risks`, and target `worldview-shift sentence`.
11. For main-manuscript hardening and other paper-facing explanation rounds, extract the `claim floor`, `claim ceiling`, and `evidence boundary` before revising text. Do not start with caveat-hardening before the supported discovery sentence is explicit.
12. For any paper-facing explanation round, run a sentence-energy pass before drafting or approving prose: identify noun-stack hotspots, missing causal glue, and unnatural formal diction.
13. For local high-salience manuscript rounds, write one `editor readout sentence` and cite at least one macro `SV#` exemplar, at least one micro sentence-craft `SV#` exemplar, and the closest architecture `SV#` exemplar from `docs/governance/scientific-voice-guide.md` before asking for a rewrite.
14. For any other paper-facing explanation round, cite at least one micro sentence-craft `SV#` exemplar when the work touches legends, captions, review-note prose, or analysis summaries that may later flow into the paper.
15. If a round changes section order, section bridges, title/abstract/introduction/discussion together, or discovery-versus-tool weight but is still scoped as `local-salience`, reject the packet as misclassified and re-route it with `Architecture scope: cross-section` or `whole-manuscript`.
16. Summarize only the task-relevant conversation history.
17. Use `Context mode: summary-only` by default.
18. Upgrade to `Context mode: summary+fork_context` only when exact wording, multi-turn decisions, or non-compressible constraints matter to the child task.
19. If delegating, after spawning a child agent, monitor its status and latest output before deciding on interruption, redirect, or shutdown.
20. Define review, handoff, and escalation requirements.
21. Keep the human at milestone approval boundaries unless the task requires earlier intervention.

## High-risk rounds

Use an explicit `implementer + reviewer + verifier` round when the task can change manuscript claims, review verdicts, closeout state, or any other paper-facing acceptance signal that could be overstated in a final report.

For these rounds, the parent must:

- classify the round explicitly as `high risk` or `not high risk` before closeout
- assign an implementer responsible for the bounded change
- name the review owner responsible for critique against the stated acceptance surface, or record an explicit compression rationale if ownership must be compressed
- name the verifier owner responsible for checking the implemented state against the packet, revised text, anchors, and any required executable or visual evidence, or record an explicit compression rationale if ownership must be compressed
- require scope-downgrade disclosure when a child cannot complete the requested surface and instead returns a narrower result
- reject closeout language that upgrades a partial implementation, partial review, or missing verification into full completion

Parent truth-maintenance checks for high-risk rounds:

- verify that the child closeout distinguishes `Plan completion`, `Review verdict`, and `Verification verdict` as separate fields
- verify that any claimed manuscript change includes the exact revised text or an explicit statement that no text was changed
- verify that before/after anchors are present when prose or asset content was changed
- verify that architecture-sensitive rounds report whether the protagonist, pivot, tool role, and worldview shift actually landed instead of assuming sentence polish was enough
- verify that whole-manuscript architecture rounds also carry an `Architecture evidence map` tying the intended spine to title, abstract opening, intro ending, pivot sentence, discovery cash-out sentence, and Discussion opening
- verify that unresolved promised joints are listed when the round leaves any requested linkage, follow-through, or hardening step incomplete
- verify that the risk classification, named owners, or explicit compression rationale are present
- verify that a verifier actually ran in verifier mode instead of restating the implementer closeout

Required closeout ledger fields for high-risk rounds from `docs/agent-ops/ROUND_CLOSEOUT_TEMPLATE.md`:

- `Risk level:`
- `Plan items owned:`
- `Plan completion:`
- `Delivered items:`
- `Deferred or dropped items:`
- `Architecture verdict:`
- `Protagonist preserved:`
- `Pivot landed:`
- `Tool role preserved:`
- `Worldview shift explicit:`
- `Architecture evidence map:`
- `Unresolved promised joints:`
- `Scope-downgrade disclosure:`
- `Delivery evidence:`
- `Review verdict:`
- `Verification verdict:`
- `Parent closeout statement:`

When `Deferred or dropped items` is non-empty, require an explicit scope downgrade instead of treating the round as fully complete.

## Reviewer-stack routing

For paper-facing work, treat `docs/agent-ops/NATURE_REVIEWER_STACK.md` as the canonical reviewer-lens source.
Do not invent ad hoc reviewer personas when the stack already covers the risk.

Use the minimal reviewer subset that matches the acceptance surface:

- `handling-editor-scope reviewer` and `reviewer-routing reviewer` for editorial fit, paper-level framing, or likely reviewer-community routing
- `cross-disciplinary-readability reviewer`, `narrative-flow reviewer`, and `cognitive-load reviewer` for paper prose, whole-paper flow, reader-burden risk, or sentence-level friction from passive phrasing, nominalization, noun stacking, or other paper-facing explanation surfaces
- use the same trio to judge protagonist stability, pivot clarity, and tool-vs-discovery weighting on high-salience manuscript rounds
- `physical-mechanism reviewer`, `acoustics-doa reviewer`, `sparse-inverse-problem-comparator reviewer`, and `statistics-evidence reviewer` for interpretation, plausibility, comparator fairness, or evidence sufficiency risk
- `figure-science-readability reviewer` for figure science, panel logic, caption burden, or main-vs-supplementary judgment

When writing a child packet or review request, name:

- the selected reviewer roles
- the acceptance surface they are judging
- the in-scope and out-of-scope surfaces they must and must not treat as review failures
- any follow-up owner if a reviewer finding must be routed to another skill

Parent acceptance on reviewer-stack use:

- the packet cites the canonical reviewer-stack doc
- the reviewer subset is minimal and task-matched
- the acceptance surface is explicit rather than implied
- paper-facing packets explicitly state the scientific-inference-over-manuscript-management rule, not only active voice, simple causality, and noun-stack-friction
- local high-salience manuscript packets explicitly state the `old-world belief`, `new-world belief`, `paper protagonist`, `pivot`, `tool role`, and `discussion worldview-shift sentence`
- `cross-section` and `whole-manuscript` packets explicitly state `Architecture scope`, the `Paper spine map`, `Results section jobs`, `Discovery cash-out section`, `Discovery-vs-tool weight budget`, and `Redundancy / breathing risks`
- main-manuscript packets identify the closest macro `SV#` exemplar and the closest micro sentence-craft `SV#` exemplar when the round touches title, abstract, Results opening, transitions, section titles, or the first paragraph of Discussion
- paper-facing explanation packets outside the main manuscript identify at least one closest micro sentence-craft `SV#` exemplar when the round touches legends, captions, review-note prose, or analysis summaries that may later flow into the paper
- review findings are consolidated at the parent layer instead of left as disconnected comments

Reviewer qualification gate:

- Reject planner or reviewer outputs as under-scoped when they do not engage the required acceptance surface.
- Reject main-manuscript-hardening review as unqualified when it ignores scientific inference versus manuscript-management language, even if it comments on grammar or passive voice.
- Exclude submission metadata placeholders from scientific-narrative review unless submission packaging is explicitly in scope.

Manuscript-hardening planning and review checklist:

- claim floor: is the strongest supported discovery sentence explicit, early, and easy to retain
- old-world belief: what default intuition must the paper replace
- new-world belief: what updated understanding should remain after one pass
- paper protagonist: is the phenomenon or organizing principle stable across title, abstract, Results, and Discussion
- pivot: where does the reader's model actually change
- tool role: does the tool reveal, preserve, or test the discovery rather than become the discovery
- weight discipline: does discovery carry more narrative mass than tool validation
- section jobs: do the Results sections each have a clear role inside one cognitive shift rather than one more item in an experiment log
- discovery cash-out: where does the paper-level discovery become unavoidable rather than optional
- redundancy / breathing: where does the manuscript repeat explanation without upgrading understanding, or stay uniformly dense enough to flatten the pivot
- claim ceiling: is the stronger unsupported interpretation clearly separated rather than implied
- evidence boundary: are true scope limits named without collapsing the paragraph into self-negation
- disciplinary narrative shift: does the prose drift from scientific inference into explanation of paper positioning or process
- defensive tone: does the text rely on `X rather than Y` framing or rebuttal-like self-defense
- self-diminishing triggers: do phrases such as `without upgrading`, `descriptive rather than`, `remains positive`, or abstract endings led by `pathway` or `constraint` lower the claim floor without adding precision
- sentence friction: do noun stacks, fact clusters, static verbs, or formal-register wording force the reader to decode syntax before the science
- causal glue: when the prose gives numbers or contrasts, does it also tell the reader what those numbers mean
- structural pacing: do section and paragraph endings carry scientific consequence rather than administrative wrap-up
- figure-as-actor phrasing: are figures or panels narrating sentences that should be carried by observations, interventions, or mechanisms
- supplement and legend leakage: do supplementary text, legends, or inline legend prose slip into manuscript-management language
- audience expectation mismatch: will physicist, acoustics, and ML readers infer different scope or evidence promises from the current wording

## Plan mode

- Keep the same parent-orchestrator routing in Plan mode.
- The top-level agent may work directly or use child agents in Plan mode for planning, exploration, checking, and review.
- Keep direct and delegated work non-mutating and plan-safe until execution mode.

## Delegation decision

Choose direct execution when the work is bounded, delegation would not improve correctness or review separation, and the round's risk controls still remain intact.

If delegating, use a single child only when the request fits one core skill, one main output bundle, and one bounded acceptance surface.

Decompose into multiple child tasks when the request:

- spans multiple skills or specialist roles
- mixes execution with a separate review, audit, or red-team pass
- contains parallelizable subproblems with different source-of-truth sets or outputs
- would force one child packet to satisfy multiple independent acceptance criteria

## Supervision loop

- After spawning a child agent, either continue non-overlapping parent work or check the child's progress explicitly.
- Before interrupting or closing a child agent, inspect its current status or latest output first.
- Close a child agent only after completion, user cancellation, clear supersession, or an explicit redirect decision grounded in observed status.
- Do not close a child agent solely because elapsed time feels long.

Quick routing defaults:

- "what is this figure or panel showing", "does this figure support the claim", or "what gap exists between this figure and this critique" -> `paper-asset-review`
- "which factors matter most", "what can we compute now", "which metric explains performance better", or "can we do a factor audit" -> `experiment-results`
- "write this as manuscript prose", "explain this for cross-disciplinary readers", or "rewrite this in plain language without losing rigor" -> `paper-submission`

## Guardrails

- Do not drift into a code-first coordination model.
- Do not create new roles when the role catalog already fits.
- Do not add new governance layers before proving a concrete workflow gap.
- Prefer manuscript objectives over implementation-centric decomposition.
- Do not delegate by reflex when direct execution is the simpler bounded path.
- Do not let direct execution erase required reviewer or verifier separation.
- Do not omit `Relevant conversation context` from a child handoff.
- Do not use `summary+fork_context` when `summary-only` is sufficient.
- Do not interrupt or close a child agent without first checking status or latest output.
