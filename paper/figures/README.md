# Figures

This directory is the final paper-facing asset surface for the active paper. It is also the literal manuscript-facing delivery surface for the six main-paper figure assets.
It should contain only:

- `Figure-Legends.md`
- `README.md`
- the six active manuscript figure JPGs
- tracked `*.layout.json` sidecars for the composed figures

Do not place generator composites, `pdf/tiff` derivatives, or manual support panels here.
Committed support panels belong in the active panel layer under `figures/output/*_panels/`.

Guidelines:
- Keep intermediate/generated artifacts under `results/<run_name>/`.
- Keep only final paper-facing assets here with stable, descriptive, sortable names (for example `fig01_paradigm-shift.jpg`, `fig04_solver-dynamics.jpg`, `fig06_universality.jpg`).
- For the active six-figure manuscript, naming is governed by `paper/manuscript/FIGURE_NAMING_CONTRACT.md`.
