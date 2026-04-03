# Closeout Integrity Contract

Use this contract when reporting task completion, review status, acceptance status, or closeout for governance, manuscript, figure, or submission work.

## Applies to

- parent closeout messages and completion claims
- task packets, review requests, and acceptance reporting
- governance and manuscript hardening rounds
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
- closeout-sensitive rounds are explicitly classified as high risk or not high risk
- high-risk rounds separate implementer, reviewer, and verifier unless an explicit non-high-risk rationale is recorded
- final completion claims are backed by independent verification rather than a single actor's summary
