# Experiment Contract

Use this contract for experiment execution, validation, and results commits.

## Applies to

- training runs
- evaluation runs
- smoke tests and functional tests
- results commits
- scripts or library changes that are exercised by a run

## Core rules

- Every results commit must be executable, tested, and reproducible.
- Every results commit must include the exact executed code and the produced artifacts together.
- Real data is mandatory for validation.
- Outputs must live under `results/<run_name>/`.
- Fail-fast behavior is preferred over silent fallback.

## Required outputs

- run logs
- manifests and fingerprints where applicable
- key metrics
- code-state or equivalent provenance
- commit message analysis and reproduction instructions

## Acceptance criteria

- smoke test completed on real data or documented real subset
- functional test completed on real data or documented real subset
- artifacts are stored under `results/<run_name>/`
- commit message documents background, purpose, setup, results, analysis, and reproduction

## Executable gates

- repository-specific experiment checks
- `scripts/paper/verify_provenance.py` for paper-facing provenance
- branch constitution in `AGENTS.md` for detailed requirements
