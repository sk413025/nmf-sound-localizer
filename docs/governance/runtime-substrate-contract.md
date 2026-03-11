# Runtime Substrate Contract

Use this contract for active code assets that still matter operationally but are not the primary identity of the `paper/nature-comm` branch.

## Applies to

- maintained runtime modules under `nmf_localizer/`
- maintained runtime modules under `doa_rl/`
- maintained runtime scripts under `scripts/` outside `scripts/paper/`
- `tests/`
- package metadata such as `pyproject.toml`, `setup.py`, `requirements.txt`, and `MANIFEST.in`

## Active core

The active runtime substrate is intentionally narrow:

- transfer-function estimation and STFT-unified preprocessing
- USM training support
- `DoADataset` and soft-OMP support
- minimal conversion and evaluation scripts still needed for current paper work

The following are not active runtime core and must remain quarantined unless explicitly reactivated:

- legacy pipeline/localizer/evaluator package surfaces
- DT and trajectory generation paths
- oracle, teacher-alignment, and broken eval paths
- research-only visualization and analysis scripts
- heavy reconstruction and fixed-dataset test suites

## Core rules

- Runtime substrate exists to support manuscript, evidence, experiment, and review work.
- Runtime substrate must not become the main branch entrypoint or override manuscript-first governance.
- Changes to runtime substrate must be justified by a current paper-facing, experiment-facing, or evidence-facing need.
- Package metadata must describe only real, maintained modules. Do not declare CLI or package surfaces that do not exist.
- Code outside the active core must move to `legacy/runtime/` or `legacy/tests/` instead of remaining ambiguously active.
- Runtime substrate outputs must follow branch output rules and avoid polluting the repository root.
- If a runtime path, default asset path, import boundary, or CLI surface changes, update the affected call sites and note the compatibility impact.

## Required outputs

- clear statement of why the runtime change is needed now
- updated path or compatibility references when runtime inputs move
- targeted validation for the touched runtime path
- evidence linkage if the runtime change supports manuscript or submission work
- explicit quarantine note when a runtime path exits the active core

## Acceptance criteria

- runtime substrate is discoverable from governance, but not treated as branch identity
- active runtime surface is limited to TF + USM + soft-OMP support paths
- branch entrypoints route to contracts and paper workflows before code modules
- moved or renamed runtime inputs have updated active call sites
- root-level asset boundaries remain clean after the change

## Executable gates

- `python scripts/paper/check_governance_links.py`
- `python scripts/paper/check_asset_boundaries.py`
- `make paper-check`
