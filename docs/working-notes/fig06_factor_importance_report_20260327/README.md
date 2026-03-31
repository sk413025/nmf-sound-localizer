# Fig. 6 Factor Importance Report

This directory contains an internal decision note that turns the current
cross-material factor audit into a self-contained illustrated report.

The report is written so that an interdisciplinary scientist, including a
reader outside the immediate subfield, can follow the logic without already
knowing the Fig. 6 implementation details.

## Scope

- audience: internal decision making
- posture: strict evidence only
- language: English, because this is repo-tracked content
- main question:
  besides low rank, which factors currently appear to control direction-encoding
  quality across everyday objects?
- excluded from the main answer:
  - a final universal equation claim
  - Top-1-driven factor ranking
  - causal percentage decomposition across factors

## Contents

- `fig06_factor_importance_report_20260327.md`: main report
- `generate_assets.py`: copies context figures and builds report-specific visuals
- `factor_tier_summary.csv`: current family-level tier summary used in the report
- `report_numbers.json`: baseline and tier numbers used by the report
- `assets/`: local copies of all images referenced by the report

## Build

From the repository root:

```bash
python docs/working-notes/fig06_factor_importance_report_20260327/generate_assets.py
```

## Evidence Anchors

- factor audit:
  `results/fig06_universal_equation_factor_audit_20260327_173647`
- cross-material geometry audit:
  `results/fig06_cross_material_geometry_20260327_171243`
- current manuscript-facing Fig. 6 review bundle:
  `figures/review_artifacts/fig06`

The local `assets/` directory copies the specific figure previews and result
plots used by this note so the Markdown report remains self-contained.
