# Closeout Integrity Contract

Use this contract when reporting task completion, review status, acceptance status, or closeout for governance, manuscript, figure, or submission work.

## Applies to

- parent closeout messages and completion claims
- task packets, review requests, and acceptance reporting
- governance and paper-facing hardening rounds
- reviewer-pass, verification, and milestone status decisions

## Core rules

- Keep closeout manuscript-first. Completion reporting exists to protect scientific quality, evidence integrity, and governance integrity rather than to accelerate superficial task closure.
- Distinguish review status, implementation status, verification status, and plan completion. A reviewer pass does not by itself establish implementation completeness or full plan completion.
- Do not report a task as complete when the accepted surface has been narrowed, partially deferred, or reframed unless that downgrade is disclosed explicitly in the closeout.
- If scope, acceptance surface, or promised outputs changed after the task packet was written, state what changed, why it changed, and which original commitments remain open.
- Closeout claims about text must preserve a hard boundary between exact text evidence and high-level interpretation.
- When claiming that language was changed, fixed, removed, added, or aligned, cite the exact text or exact diff evidence rather than only line numbers, file summaries, or paraphrase.
- Line references may support navigation, but they do not replace the underlying text evidence.
- High-level interpretation must be labeled as interpretation and must not be presented as if it were the exact manuscript or governance text.
- When a high-risk prose or governance round claims architectural improvement, the closeout must say whether the intended protagonist, pivot, tool role, and worldview shift actually landed.
- When broader significance is in scope for that round, the same closeout must also say whether any promised second-layer discovery actually landed.
- When a whole-manuscript or cross-section hardening round claims architectural improvement, the closeout must include an `Architecture evidence map` tying title, the `front-door preload sentence` when one is required, pivot sentence, discovery cash-out sentence, and Discussion opening to the intended spine.
- When a round claims broader significance landed, the closeout must distinguish the `second-layer discovery` from any downstream consequence and must report whether the `broader-implication trunk` and `no-bolt-on test` landed.
- When a round claims broader significance landed, the closeout must also report the earned promotion level and any demotion that review or verification required.
- For any `high-risk` round with broader significance or cross-disciplinary consequence in scope, the closeout must be backed by `results/<round_name>/governance_round.yaml` and a passing `make paper-governance-gate ROUND_DIR=results/<round_name>` run.
- Classify each closeout-sensitive round as high risk or not high risk before assigning reviewer and verifier ownership.
- For high-risk rounds that can change manuscript claims, governance posture, or acceptance status, separate implementer, reviewer, and verifier roles.
- In a high-risk round, a reviewer pass is advisory to acceptance until an independent verifier confirms that the implemented state matches the claimed closeout.
- The verifier must check the actual changed text or diff evidence and must not rely only on the implementer's summary, the reviewer's summary, or a parent synthesis.
- If one person or one agent necessarily performs more than one role, the parent must record the constraint and explain why the round is not being treated as high risk or why temporary role compression is unavoidable.

## Required outputs

- closeout that states whether it is reporting review status, implementation status, verification status, plan completion, or a combination
- explicit disclosure of any scope downgrade, deferral, or remaining gap
- exact text or diff evidence for text-facing completion claims
- separate high-level interpretation when interpretation is useful
- independent verification record for high-risk completion claims

## Acceptance criteria

- closeout does not equate reviewer pass with plan completion
- closeout does not hide scope reduction behind summary language
- exact text evidence is available for text-facing claims
- high-level interpretation is clearly distinguished from exact text evidence
- architecture-sensitive rounds do not imply protagonist, pivot, worldview-shift, or second-layer-discovery success without naming those verdicts explicitly
- broader-significance rounds do not imply that the trunk or branch was earned without naming the promotion level or demotion outcome
- architecture-sensitive rounds do not imply whole-paper landing without an `Architecture evidence map`
- closeout-sensitive rounds are explicitly classified as high risk or not high risk
- high-risk rounds separate implementer, reviewer, and verifier unless an explicit non-high-risk rationale is recorded
- final completion claims are backed by independent verification rather than a single actor's summary
- broader-significance closeout distinguishes `second-layer discovery` from `downstream consequence` rather than collapsing them into one vague implication claim
- `high-risk` broader-significance closeout is backed by a machine-readable `governance_round.yaml` artifact whose final status and demotion outcome match review and verification
