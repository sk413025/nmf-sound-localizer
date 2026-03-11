# Submission Contract

Use this contract for Nature Communications compliance, figure/table packaging, source-data expectations, and final submission-facing assets.

## Applies to

- figure export and validation rules
- tables and metadata
- submission package requirements
- data and code availability sections
- paper-facing asset review

## Core rules

- The canonical Nature Communications requirements file is the source of truth.
- Legacy local figure notes are redirects, not authority.
- Paper-facing asset review must stay manuscript-first.
- When local implementation conflicts with current Nature guidance, update the canonical document and implementation together.

## Required outputs

- submission-ready manuscript assets under `paper/`
- validated figure outputs and review artifacts
- required submission sections and metadata
- explicit handling of data and code availability

## Acceptance criteria

- paper asset review workflow has been run when applicable
- manuscript includes the required branch-level submission sections
- figure files and references are consistent with branch policy
- canonical Nature requirements remain reachable from main entrypoints

## Executable gates

- `make paper-review-assets`
- `make paper-review-gate`
- `python scripts/paper/check_figure_references.py`
- `python scripts/paper/check_governance_links.py`
