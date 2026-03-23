# Nature Communications manuscript workspace

This folder is a **Markdown-first** manuscript workspace intended for drafting and iterating quickly, then exporting to journal submission formats via **Pandoc**.

## Layout
- `paper/manuscript/`: Manuscript source (`manuscript.md`) + Pandoc metadata (`metadata.yaml`).
- `paper/references/`: Bibliography (`references.bib`).
- `paper/figures/`: Final, submission-ready figure assets (PDF/TIFF/PNG as needed).
- `paper/templates/`: Pandoc `reference.docx` controlling DOCX styling.
- `paper/csl/`: CSL file (`style.csl`) controlling citation style.
- `paper/out/`: Build outputs (ignored by git).

## Build (DOCX)
Prerequisite: `pandoc` (tested with Pandoc 3.x).

```bash
make paper-build
```

Outputs:
- `paper/out/manuscript.docx`
- `paper/out/build.log`

## Governance and checks

This manuscript workspace sits inside a branch-level governance system.

- Branch constitution: `AGENTS.md`
- Governance contracts: `docs/governance/`
- Human and agent quickstarts: `START_HERE_HUMAN.md`, `START_HERE_AGENT.md`

Recommended health check:

```bash
make paper-check
```

## Nature Communications formatting (submission)
Nature Communications is typeset in two columns at publication, but Word submissions should be formatted for review readability (e.g., double-spaced, single-column, no full justification, with Arabic page numbers). The journal does not provide a Word template; formatting is imposed later during production.

This repo approximates the submission style via:
- `paper/templates/reference.docx`: Pandoc reference DOCX (page size/margins, double spacing, footer page numbers).
- `paper/csl/style.csl`: Numeric citation style (Nature-like).

If you want line numbers for review, enable them in Word (Layout → Line Numbers → Continuous).

## Templates and citation style (customization)
This repo includes a reasonable default `reference.docx` and a numeric CSL style. If you have a journal-provided Word template or a house citation style, replace the files below:
- DOCX reference document: `paper/templates/reference.docx`
- CSL style: `paper/csl/style.csl`

If either file is missing, the build still succeeds using Pandoc defaults, and prints a note to stderr.

## Figures
Keep figure generation runs under `results/<run_name>/` (existing figure scripts already follow this convention in this repo).
When a figure is ready for manuscript submission, copy the final asset into `paper/figures/` with a stable name.
Recommended convention: `fig01_<short-topic>.<ext>` (e.g., `fig02_svd-physical-dictionary.jpg`).
The active six-figure naming contract is tracked in `paper/manuscript/FIGURE_NAMING_CONTRACT.md`, and intended panel sizes are tracked in `figures/conf/layout_spec.md`.
