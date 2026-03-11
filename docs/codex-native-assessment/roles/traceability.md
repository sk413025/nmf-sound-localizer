# Role Packet: Evidence, Provenance, and Traceability Agent

## Objective

Evaluate whether Codex can reliably connect scientific claims to figures, artifacts, and provenance in this branch.

## Required inputs

- `docs/codex-native-assessment/CODEX_CAPABILITY_BASELINE.md`
- `docs/codex-native-assessment/SHARED_RUBRIC.md`
- `figures/FIGURE_REGISTRY.md`
- `paper/manuscript/manuscript.md`
- `paper/references/CLAIM_CITATION_MATRIX.md`
- `scripts/paper/resolve_tbd.py`
- `scripts/paper/verify_provenance.py`

## Focus

- claim-to-figure linkage
- figure-to-artifact linkage
- manuscript placeholder provenance
- machine-readable evidence flow for future agents

## Questions to answer

- Can each major manuscript claim be traced to a figure and to a concrete artifact path?
- Are provenance checks sufficient for agent-assisted paper work?
- Which missing links are currently forcing agents to guess?
- Should the branch add an evidence manifest for Codex use?

## Required outputs

- a traceability gap map
- the most dangerous evidence breaks for agent-assisted writing
- recommended evidence manifests or indexes
- acceptance criteria for future provenance checks

## Failure conditions

Your report should be considered weak if it lists data files without connecting them to manuscript claims or if it ignores the existing provenance tooling.
