# Asset Classes

This branch treats every tracked file, directory, script, and artifact as a governed project asset. Every asset must belong to one of the classes below.

## Canonical paper assets

These assets directly support the current Nature Communications manuscript workflow and may appear in branch entrypoints, contracts, skills, and paper-facing checks.

Examples:

- `paper/`
- `figures/`
- `scripts/paper/`
- `docs/governance/`
- `docs/agent-ops/`
- `.codex/skills/`
- paper-facing provenance inputs and outputs under `results/`

## Active runtime substrate

These assets remain active and executable, but they are not the identity of the branch. They may support paper work, experiment work, or evidence generation, but they must not replace manuscript-first governance.

Examples:

- `nmf_localizer/`
- `doa_rl/`
- `scripts/` outside `scripts/paper/`
- `tests/`
- `pyproject.toml`
- `setup.py`
- `requirements.txt`
- `MANIFEST.in`

This class is governed by `runtime-substrate-contract.md`.

## Evidence docs

These documents preserve evidence, lineage, provenance, or artifact context that may still support paper decisions. They may be referenced when evidence is needed, but they are not branch entrypoints.

Examples:

- `docs/evidence/`

## Working notes

These documents capture analysis, prompts, specs, investigations, and exploratory reasoning. They are useful project assets, but they are not canonical source of truth and must not drive branch-level decisions by themselves.

Examples:

- `docs/working-notes/`

## Quarantined legacy assets

These assets remain in the repository for historical, operational, or recovery reasons, but they are intentionally outside the main branch workflow.

Examples:

- `legacy/assets/`
- `legacy/scripts/`

## Governance rule

If an asset does not clearly belong to one of these classes, it must be reclassified, moved, or deleted before it can remain in active use.
