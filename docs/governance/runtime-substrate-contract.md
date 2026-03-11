# Runtime Substrate Contract

Use this contract for active code assets that still matter operationally but are not the primary identity of the `paper/nature-comm` branch.

## Applies to

- `nmf_localizer/`
- `doa_rl/`
- `scripts/` outside `scripts/paper/`
- `tests/`
- package metadata such as `pyproject.toml`, `setup.py`, `requirements.txt`, and `MANIFEST.in`

## Core rules

- Runtime substrate exists to support manuscript, evidence, experiment, and review work.
- Runtime substrate must not become the main branch entrypoint or override manuscript-first governance.
- Changes to runtime substrate must be justified by a current paper-facing, experiment-facing, or evidence-facing need.
- Runtime substrate outputs must follow branch output rules and avoid polluting the repository root.
- If a runtime path, default asset path, import boundary, or CLI surface changes, update the affected call sites and note the compatibility impact.

## Required outputs

- clear statement of why the runtime change is needed now
- updated path or compatibility references when runtime inputs move
- targeted validation for the touched runtime path
- evidence linkage if the runtime change supports manuscript or submission work

## Acceptance criteria

- runtime substrate is discoverable from governance, but not treated as branch identity
- branch entrypoints route to contracts and paper workflows before code modules
- moved or renamed runtime inputs have updated active call sites
- root-level asset boundaries remain clean after the change

## Executable gates

- `python scripts/paper/check_governance_links.py`
- `python scripts/paper/check_asset_boundaries.py`
- `make paper-check`
