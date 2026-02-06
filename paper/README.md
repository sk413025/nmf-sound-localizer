# Nature Communications manuscript workspace

This folder is a **Markdown-first** manuscript workspace intended for drafting and iterating quickly, then exporting to journal submission formats via **Pandoc**.

## Layout
- `paper/manuscript/`: Manuscript source (`manuscript.md`) + Pandoc metadata (`metadata.yaml`).
- `paper/references/`: Bibliography (`references.bib`).
- `paper/figures/`: Final, submission-ready figure assets (PDF/TIFF/PNG as needed).
- `paper/templates/`: Optional Pandoc `reference.docx` template for journal styling.
- `paper/csl/`: Optional CSL file (`style.csl`) controlling citation style.
- `paper/out/`: Build outputs (ignored by git).

## Build (DOCX)
Prerequisite: `pandoc` (tested with Pandoc 3.x).

```bash
./scripts/paper/build_docx.sh
```

Outputs:
- `paper/out/manuscript.docx`
- `paper/out/build.log`

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
