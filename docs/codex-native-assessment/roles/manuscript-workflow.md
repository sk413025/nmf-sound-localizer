# Role Packet: Manuscript Workflow Agent

## Objective

Evaluate whether this branch is organized for Codex-assisted manuscript drafting, revision, and consistency control.

## Required inputs

- `docs/codex-native-assessment/CODEX_CAPABILITY_BASELINE.md`
- `docs/codex-native-assessment/SHARED_RUBRIC.md`
- `AGENTS.md`
- `paper/manuscript/manuscript.md`
- `paper/manuscript/REVISION_CHECKLIST.md`
- `figures/FIGURE_REGISTRY.md`

## Focus

- manuscript-first source-of-truth clarity
- section-level task decomposition
- claim-to-evidence writing workflow
- placeholder and consistency management
- suitability for parallel manuscript work

## Questions to answer

- Can a fresh Codex agent take a section-level manuscript task without reading the whole repository?
- Are claims, figures, and Methods consistency easy to audit?
- Should parts of the writing workflow move into machine-readable manifests or reusable skills?
- Does the current branch structure make manuscript work easier than figure work?

## Required outputs

- a workflow map of the current manuscript process
- the top blockers to Codex-assisted writing
- recommended manuscript-native interfaces or documents
- ranked recommendations with acceptance criteria

## Failure conditions

Your report should be considered weak if it focuses only on figure files, ignores `AGENTS.md`, or does not connect recommendations back to manuscript quality.
