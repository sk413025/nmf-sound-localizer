# Cross-Material Progress Deck

This directory contains a short external-facing progress deck for the March 24-25, 2026 cross-material white-noise results.

## Scope

- audience: external academic talk
- duration: about 10 minutes
- language: English slides
- main story: five-material cross-material progress only
- excluded from main flow: `speech20`

## Contents

- `cross_material_progress_20260327.md`: Pandoc slide source
- `speaker_notes.md`: slide-by-slide talk track
- `generate_assets.py`: reproducible asset builder for the workflow slide
- `build_deck.sh`: builds the `.pptx`

## Build

From the repository root:

```bash
bash docs/working-notes/cross_material_progress_deck_20260327/build_deck.sh
```

Output:

- `paper/out/cross_material_progress_20260327.pptx`

## Evidence Anchors

- cross-material processing record:
  `../ldv-master-reference-audio/results/cross_materials_material_selection_20260324_221916`
- Fig. 6 support bundle:
  `../ldv-master-reference-audio/results/cross_materials_fig06_support_20260325_164051`
- current manuscript provenance anchor:
  `DATA_PROVENANCE.md`

The deck reuses the current Fig. 6 review-bundle previews for readability and keeps the main message aligned with the manuscript-facing wording.
