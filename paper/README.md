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

## Templates and citation style (optional)
- If you have a journal Word template (or a house style reference document), place it at:
  - `paper/templates/reference.docx`
- If you have a CSL style file, place it at:
  - `paper/csl/style.csl`

If either file is missing, the build still succeeds using Pandoc defaults, and prints a note to stderr.

## Figures
Keep figure generation runs under `results/<run_name>/` (existing figure scripts already follow this convention in this repo).
When a figure is ready for manuscript submission, copy the final asset into `paper/figures/` with a stable name.

