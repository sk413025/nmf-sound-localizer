---
lang: en-US
---

# Cross-Material Progress

**From Reproduction to Screening Across Five Everyday Objects**

Progress snapshot: March 24-25, 2026  
White-noise only, external 10-minute update

![](../../../figures/review_artifacts/fig06/panel_previews/a.png){width=90%}

# Question and Setup

- **Question:** do direction-frequency encoders reproduce across materials and remain useful for downstream screening?
- **Input bundle:** `5 materials x 37 angles = 185` recordings from `cross-materials.zip`
- **Scope today:** executed white-noise chain only; speech results are intentionally excluded

![](assets/workflow_overview.png){width=95%}

# Cross-Material H Reproduces

- Structured angle-frequency encoding is visible in every material under shared normalization.
- This is breadth evidence: the encoder is not confined to one acrylic-like object.
- Mean coherence is informative for physics inspection, but it is **not** the selection criterion by itself.

![](../../../figures/review_artifacts/fig06/panel_previews/b.png){width=95%}

# Low-Rank Structure Persists Across Materials

- `rank95` stays between `3` and `6` modes across the five materials.
- Low-rank physical structure therefore survives cross-material transfer rather than collapsing into material-specific noise.
- This extends the physical-dictionary view beyond the single-object case.

![](../../../figures/review_artifacts/fig06/panel_previews/c.png){width=93%}

# Screening Outcome: B Primary, W Backup

- `B` is the current primary target; `W` is the strongest backup.
- Key downstream numbers:
  - `B`: Top-1 `0.811`, within-10deg `0.955`, MAE `2.61 deg`
  - `W`: Top-1 `0.793`, within-10deg `0.973`, MAE `5.09 deg`
- Important negative result: `P` has the strongest H-level coherence, but it does **not** win on task.

![](../../../figures/review_artifacts/fig06/panel_previews/d.png){width=95%}

# Why Screening Is Not Just Energy

- Overall response energy does not explain screening outcome by itself.
- What separates materials is frequency-localized directional structure, not just "more signal."
- That is why the cross-material story stays on physical encoding plus screening, not raw amplitude alone.

![](../../../figures/review_artifacts/fig06/panel_previews/e.png){width=95%}

# Status, Limits, and Next Steps

- **Status**
  - audit complete
  - legacy ingest complete
  - `Original -> Material` H reproduced
  - downstream screening complete
  - Fig. 6 support bundle assembled
- **Current limit**
  - the ZIP passes the `5 x 37 = 185` inventory check, but strict raw rerecord compliance is not yet verifiable from workspace evidence alone
- **Next step**
  - carry `B` forward first, keep `W` as backup, and expand cross-material validation without over-claiming solver superiority

# Backup: Full Screening Metrics

| Material | Top-1 | Within 10deg | MAE deg | Mean coherence |
|:--|--:|--:|--:|--:|
| B | 0.8108 | 0.9550 | 2.61 | 0.0274 |
| W | 0.7928 | 0.9730 | 5.09 | 0.0205 |
| A | 0.7838 | 0.9369 | 3.83 | 0.0202 |
| P | 0.7748 | 0.9189 | 5.99 | 0.0332 |
| M | 0.6757 | 0.9279 | 5.23 | 0.0259 |

# Backup: Matched Greedy vs Hybrid Benchmark

This result is **not** part of the main 10-minute story. It is backup evidence only.

| Material | Greedy Top-1 | Hybrid Top-1 | Greedy MAE deg | Hybrid MAE deg |
|:--|--:|--:|--:|--:|
| B | 0.955 | 0.964 | 0.315 | 0.270 |
| W | 1.000 | 1.000 | 0.000 | 0.000 |
| A | 0.964 | 0.964 | 0.721 | 0.721 |
| P | 1.000 | 1.000 | 0.000 | 0.000 |
| M | 0.991 | 0.991 | 0.045 | 0.045 |

# Backup: Provenance Chain

Executed white-noise chain:

1. `results/cross_materials_protocol_audit_20260324_213941`
2. `results/cross_materials_legacy_stage0_20260324_215737`
3. `results/h_matrix_repro_original_to_cross_materials_20260324_220841`
4. `results/cross_materials_material_selection_20260324_221916`
5. `results/cross_materials_fig06_support_20260325_164051`

Current manuscript anchor:

- Fig. 6 in `DATA_PROVENANCE.md`
- current legend wording in `paper/figures/Figure-Legends.md`
