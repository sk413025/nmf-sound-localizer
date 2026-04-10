# Codex Collaboration Contract

Use this contract for any task about Codex-native workflow, multi-agent organization, governance design, AGENTS usage, or local skills.

## Applies to

- `AGENTS.md`
- `START_HERE_AGENT.md`
- project-local skills under `.codex/skills/`
- `docs/agent-ops/`
- `docs/agent-ops/NATURE_REVIEWER_STACK.md`
- supervisor, specialist, and red-team coordination

## Core rules

- Define `codex-native` from actual Codex capabilities and local repository primitives, not from taste alone.
- Prefer existing repository primitives such as `AGENTS.md`, local skills, and `scripts/paper/` over parallel systems.
- Prefer simplification over accretion. New governance fields, verdicts, or artifacts must clear an anti-duplication bar rather than being added by default.
- Keep collaboration manuscript-first.
- Make code subordinate to manuscript, evidence, and submission workflows.
- Keep agent stance aligned with Nature-facing scientific writing: rigorous, editor-legible, and claim-forward rather than reflexively defensive.
- Route paper-facing hardening and other high-salience prose work through `docs/governance/scientific-voice-guide.md` so sentence-shape guidance comes from one canonical exemplar set rather than from duplicated local heuristics.
- Treat sentence-level readability as part of scientific rigor, not as optional polish after claim-floor work is finished.
- Require paper-facing workflow to optimize for reader-first scientific explanation, not only for contract compliance or hedge removal.
- Treat narrative architecture as part of scientific rigor. The branch must optimize for one primary cognitive shift with one stable protagonist, and when the evidence supports broader significance it must also express one endogenous second-layer discovery rather than only a list of applications.
- Treat broader significance as a governed promotion ladder rather than an intuition call. The only allowed statuses are `core-only`, `second-layer earned`, `branch earned`, and `leaf allowed`.
- Do not let one concept acquire multiple canonical homes. If a schema, ladder, or field inventory is machine-readable in one place, other docs should reference it rather than re-implement it.
- Do not let the checker over-model reviewer judgment. Executable gates should enforce consistency and ownership boundaries, not infer higher-level paper verdicts that belong to review.
- Treat this contract as the canonical home of anti-complexity discipline for governance and workflow changes. Other docs may apply that discipline, but they should not restate a second full rule set.
- Any governance expansion must answer four questions before landing:
  - why existing primitives were insufficient
  - what duplicated surface was removed
  - what remains canonical after the change
  - what complexity risk stays open
- If a governance change adds a field, verdict, artifact, or workflow branch without also deleting or collapsing an older duplicated surface, treat the change as presumptively invalid.
- Classify `Architecture scope` before paper-facing hardening begins. Use `local-salience` only for local high-salience rewrites with no section reweighting, no section-bridge changes, and no Results-spine changes; use `cross-section` for rounds that change more than one section, section bridges, or discovery-versus-tool weight; use `whole-manuscript` for full-paper restructuring or any round that re-architects the Results spine.
- Require paper-facing workflow to identify `old-world belief`, `new-world belief`, `paper protagonist`, `pivot`, and `tool role` for high-salience manuscript rounds.
- Require whole-manuscript and cross-section hardening to write a `Paper spine map` with Results section jobs, pivot sentence, discovery cash-out section, discovery-versus-tool weight budget, second-layer discovery, broader-implication trunk, and downstream-consequence branch before local sentence polishing begins.
- Keep architecture artifacts phase-correct: `Paper spine map` is a drafting-time packet artifact, while `Architecture evidence map` is a closeout-time verification artifact for `cross-section` and `whole-manuscript` rounds that claim architecture landed.
- Do not let experiment chronology, tool fascination, or section-local optimization substitute for paper architecture.
- Treat broader significance as architecture, not as a late decorative paragraph. If the paper cannot state a supported second-layer discovery, the workflow must not compensate by adding a wider application inventory.
- Require broader-significance promotion to pass three named tests before the parent may let it into the paper spine: `Earned-discovery test`, `Boundary-pressure test`, and `Reviewer-routing survival test`.
- Require the workflow to demote broader significance when those tests fail. If the trunk fails, demote to `core-only`; if a branch fails, demote to `second-layer earned`; if a leaf fails, drop it or demote to `branch earned`.
- For any `high-risk` round with broader significance or cross-disciplinary consequence in scope, require one canonical machine-readable artifact at `results/<round_name>/governance_round.yaml`.
- Treat `governance_round.yaml` as the blocking source of truth for promotion level, review-forced demotion, verification status, and closeout-ready status on those rounds. Human-readable memos may accompany it, but they do not replace it.
- Distinguish branch-local source of truth from archive material.
- Route the top-level agent through `agent-orchestrator` and require an explicit execution-or-delegation decision before specialist work begins.
- Multi-agent recommendations must include explicit acceptance criteria and ownership boundaries.
- If delegation is chosen, the top-level agent must make an explicit decomposition decision before execution: single child only for genuinely single-scope work, otherwise split the request into multiple child tasks.
- Require every child-agent handoff to include task framing plus `Relevant conversation context`.
- Default child-agent handoff to `Context mode: summary-only`; use `summary+fork_context` only when exact task-relevant dialogue cannot be safely compressed.
- Do not pass irrelevant thread history, hidden reasoning, or expected answers to child agents.
- In this repository's default operating mode, treat the human as providing standing authorization for sub-agent use and let the top-level agent decide when child agents are needed.
- When delegation is used, require the top-level agent to monitor active child agents and inspect status before interrupting or closing them.
- Do not close a child agent solely because elapsed time feels long.
- For paper-facing hardening, editor-scope review, reviewer-routing review, or high-stakes red-team critique, require the supervisor to select the applicable reviewer roles and evaluation goals from `docs/agent-ops/NATURE_REVIEWER_STACK.md` rather than inventing ad hoc reviewer personas.
- Treat the canonical Nature reviewer stack as a review-lens layer routed through existing roles and skills, not as a parallel workflow system.
- Require the parent to classify closeout-sensitive rounds as high risk or not high risk before assigning reviewer and verifier ownership.
- Treat reviewer pass, implementation completion, verification completion, and plan completion as separate decisions.
- Require parent closeout to disclose any scope downgrade or deferred acceptance surface instead of implying full completion.
- For high-risk rounds that can change manuscript claims, governance posture, or acceptance status, require implementer, reviewer, and verifier separation unless the parent records why the round is not high risk.
- Parent closeout must cite exact text or exact diff evidence for text-facing claims. Line references and summaries may assist navigation, but they do not replace the underlying evidence.
- Require independent verification before the parent reports full completion on a high-risk round.
- Do not let high-risk closeout discipline bleed into paper voice. Evidence-boundary rigor belongs in routing, review, and closeout, but paper-facing explanation must still state the supported claim floor clearly.
- For paper-related figures, use Codex multimodal capability on the real asset rather than metadata-only inference.
- Require image inspection for `jpg` and `png`, and page-by-page PDF-to-PNG conversion before figure interpretation when the asset is a `pdf`.
- For generated or data-backed figures, require a three-layer check: visual asset, generator or composition code, and upstream evidence or provenance artifact.

## Required outputs

- clear routing for human, agent, and supervisor roles
- reusable skills and unified task packets for repeated work
- task packets that record `Relevant conversation context` and a `Context mode` decision
- review plans that name the applicable canonical reviewer roles when paper-facing critique is required
- packet, review, and closeout surfaces that record broader-significance status coherently and defer machine-readable status mechanics to the canonical schema
- one canonical machine-readable field inventory for `high-risk` broader-significance rounds, referenced rather than recopied across packet, reviewer, and closeout docs
- a canonical `governance_round.yaml` artifact plus executable semantic validation for `high-risk` broader-significance rounds
- evidence-backed recommendations
- closeout that distinguishes exact text evidence from high-level interpretation
- executable checks where policy is high value and low ambiguity
- a stable agent operating model for task decomposition, handoff, and review

## Acceptance criteria

- main entrypoints clearly route to the right canonical docs
- local skills align with branch governance and stay limited to the core branch workflows
- top-level routing goes through `agent-orchestrator`
- agent-first guidance makes it explicit that manuscript rigor is not implemented as timid or self-diminishing scientific prose
- agent-first guidance makes it explicit that editorial adequacy is not established by sentence polish alone when protagonist, pivot, or tool-vs-discovery weighting are still wrong
- agent-first guidance makes whole-paper architecture explicit enough that a rewriter cannot satisfy the workflow with local paragraph improvements alone
- task packets and role definitions are discoverable from the main governance path
- top-level routing guidance distinguishes direct execution from delegated execution and explains when decomposition is required
- task packets include `Relevant conversation context` and explicit `Context mode`
- supervision guidance requires child-status checks before interruption or shutdown
- the canonical Nature reviewer stack is discoverable from the main governance path and routed through existing roles rather than duplicated
- closeout guidance does not equate reviewer pass with plan completion and requires scope-downgrade disclosure
- closeout-sensitive rounds are explicitly risk-classified before reviewer and verifier ownership is assigned
- high-risk coordination requires implementer, reviewer, and verifier separation or an explicit non-high-risk rationale
- text-facing closeout claims are backed by exact text or diff evidence rather than summary alone
- governance checks confirm the key files and links exist
- Codex-native orchestration guidance remains discoverable from the main governance path
- paper-related figure workflows do not allow metadata-only acceptance when visual inspection or provenance backtrace is required
- paper-facing workflow exposes whether broader significance landed as a second-layer discovery, whether downstream consequences stayed bounded, and whether the `no-bolt-on` test passed
- paper-facing workflow also exposes whether broader significance earned its current promotion level or should be demoted
- `high-risk` broader-significance rounds cannot close out as landed without a passing `governance_round.yaml` semantic gate
- governance changes that add a field, verdict, or artifact also remove or collapse an older duplicated surface

## Executable gates

- `python scripts/paper/check_governance_links.py`
- `python scripts/paper/check_round_governance_semantics.py --round-dir results/<round_name>` for any `high-risk` broader-significance round
- `make paper-governance-gate ROUND_DIR=results/<round_name>` for any `high-risk` broader-significance round
- `make paper-check`
- `docs/agent-ops/README.md`
