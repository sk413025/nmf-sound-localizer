# Role Packet: Automation and Validation Agent

## Objective

Evaluate whether this branch provides deterministic commands and validation steps that Codex can run to assess manuscript readiness.

## Required inputs

- `docs/codex-native-assessment/CODEX_CAPABILITY_BASELINE.md`
- `docs/codex-native-assessment/SHARED_RUBRIC.md`
- `Makefile`
- `paper/README.md`
- `scripts/paper/build_docx.sh`
- `scripts/paper/resolve_tbd.py`
- `scripts/paper/verify_provenance.py`

## Focus

- command entrypoints
- paper health checks
- deterministic validation of manuscript and submission readiness
- how well the branch supports autonomous shell-based evaluation

## Questions to answer

- Is there a single command surface for paper health?
- Which checks are missing for placeholders, metadata, figure links, and provenance?
- Which validations are already scriptable and which still depend on manual judgment?
- What should a future `paper-check` command include?

## Required outputs

- a current command surface map
- the missing validation inventory
- ranked recommendations for paper-facing commands
- acceptance criteria for a manuscript CI gate

## Failure conditions

Your report should be considered weak if it proposes generic CI work without tying it back to Codex-run validation and manuscript outcomes.
