# Manuscript Contract

Use this contract for any task that changes manuscript text, figure legends, paper structure, or the scientific narrative.

## Applies to

- `paper/manuscript/manuscript.md`
- figure legends and claim framing
- Results, Discussion, and Methods boundary decisions
- manuscript consistency and placeholder cleanup

## Core rules

- The branch is manuscript-first.
- Main text structure must remain `Introduction -> Results -> Discussion -> Methods`.
- Results should be assertion-first and evidence-driven.
- Methods carries procedural detail and reproducibility-critical specifications.
- Symbols and advanced concepts must be introduced with bridge sentences and clear physical meaning.
- Paragraphs should end with evidence-anchored take-home claims.

## Required outputs

- manuscript text consistent with the branch writing style
- correct figure references
- consistent symbols across sections
- no unexplained placeholders in final paper-facing output

## Acceptance criteria

- required manuscript sections are present
- figure references are consistent with the manuscript and figure registry
- unresolved placeholders are explicitly tracked or resolved
- paper-facing checks under `make paper-check` pass

## Executable gates

- `python scripts/paper/check_required_sections.py`
- `python scripts/paper/check_figure_references.py`
- `python scripts/paper/verify_provenance.py`
- `make paper-build`
