# Master Figure 3 Lineage Crosswalk

Date: 2026-03-12

Purpose: disentangle which current figures, generator PDFs, results PDFs, and legacy assets actually descend from the original `Master Figure 3` after the manuscript was expanded from one mechanism composite into the current `Fig. 5–8` family.

Canonical provenance remains in:

- `figures/FIGURE_REGISTRY.md`
- `figures/conf/review_targets.yaml`
- `figures/conf/experiments.yaml`

## Reading rule

Each asset in this note is classified as one of:

- `panel a+c descendant`
- `panel b upper descendant`
- `panel b lower descendant`
- `combined master-figure descendant`
- `separate family, not in Master Figure 3 lineage`

The classification uses visual content first, then registry metadata, results provenance, and git history as support.

---

## 1. Original anchor: what Master Figure 3 actually is

### Canonical original asset

| Asset | Type | Visual title / content | Classification |
|---|---|---|---|
| `legacy/assets/master_figure3_reference_images/Figure-5.png` | legacy image | `Master Figure 3 (Final): Deciphering AI Understanding via Micro-Mechanism & Macro-Robustness` with three panels: global attention, micro-level mechanism, macro-level selection robustness | Original anchor |

### Original panel semantics

| Original panel | Visual content | Scientific message |
|---|---|---|
| `a` | global physics-consistent attention | the model learns a routing structure aligned with the physical angle manifold |
| `b` upper | micro-level case study with two angle-specific comparison blocks | the physics-aware model sharpens the correct selection relative to the baseline |
| `b` lower | polar estimate | old visual summary of the same case-study mechanism; no longer active |
| `c` | all-angle selection probability heatmaps | OMP spreads mass off diagonal, while the physics-aware model remains sharply diagonal |

The current manuscript no longer compresses these ideas into one composite. Instead, it expands them into a four-figure mechanism family:

- `Fig. 5`: old `a + c`
- `Fig. 6`: old `b` upper
- `Fig. 7–8`: old `b` lower replacement branch (band-wise diagnostics)

---

## 2. Direct descendants of the original panels

### Old panel `a + c` → current Fig. 5

| Asset | Type | Visual content summary | Current status | Classification | Recommendation |
|---|---|---|---|---|---|
| `paper/figures/fig05_structure-macro-selection.png` | current manuscript composite | left column: `H` and `QK` structure; right column: OMP and physics-aware all-angle selection maps | current main | panel a+c descendant | Keep as canonical active descendant |
| `figures/output/fig05_structure_alignment.pdf` | current generator PDF | full upstream split of the old global-structure and macro-robustness content | current generator output | panel a+c descendant | Keep as active upstream support |
| `figures/output/fig05_routing_mechanism_analysis_panels/fig05_panel_a_global_attention.pdf` | split panel PDF | isolated left-column structure descendant | current generator output | panel a+c descendant | Keep as panel-level provenance |
| `figures/output/fig05_routing_mechanism_analysis_panels/fig05_panel_c_macro_robustness.pdf` | split panel PDF | isolated right-column macro-robustness descendant | current generator output | panel a+c descendant | Keep as panel-level provenance |
| historical `Fig5_A1_*`, `Fig5_A2_*`, `Fig5_C1_*`, `Fig5_C2_*` | legacy panel families | early atomic decompositions of the same two branches | legacy | panel a+c descendant | Legacy only |

Interpretation:

- The old combined `a + c` story is again the canonical main-paper path.
- The confusion-heatmap replacement that briefly occupied manuscript `Fig. 5c` is no longer the active descendant.

### Old panel `b` upper → current Fig. 6

| Asset | Type | Visual content summary | Current status | Classification | Recommendation |
|---|---|---|---|---|---|
| `paper/figures/fig06_angle-specific-mechanism.png` | current manuscript composite | angle-specific probability distributions at `55°` and `100°` for baseline vs no-transformer | current main | panel b upper descendant | Keep as canonical active descendant |
| `figures/output/fig09_angle55.pdf` | current generator PDF | angle-specific distribution at `55°` | current generator output | panel b upper descendant | Keep as active upstream support |
| `figures/output/fig09_angle100.pdf` | current generator PDF | angle-specific distribution at `100°` | current generator output | panel b upper descendant | Keep as active upstream support |
| historical `Fig5_B1_*`, `Fig5_B2_*` | legacy panel families | earlier case-study renderings | legacy | panel b upper descendant | Legacy only |

Interpretation:

- The old stem/bar comparison has been updated into a cleaner probability-distribution view.
- The scientific role is unchanged: show correct-atom concentration versus off-axis leakage at representative directions.

### Old panel `b` lower replacement branch → current Figs. 7 and 8

| Asset | Type | Visual content summary | Current status | Classification | Recommendation |
|---|---|---|---|---|---|
| `paper/figures/fig07_bandwise-routing-analysis-part1.png` | current manuscript composite | full-band smoothed, full-band no-smoothing, and `300–500 Hz` routing diagnostics | current main | panel b lower descendant | Keep as active descendant |
| `paper/figures/fig08_bandwise-routing-analysis-part2.png` | current manuscript composite | `500–1000 Hz`, `1000–2000 Hz`, and `2000–3000 Hz` diagnostics | current main | panel b lower descendant | Keep as active descendant |
| `results/fig5_b3_line5_20260202_224349/Fig5_B3_LINE_300_3000.pdf` | results PDF | original smoothed full-band diagnostic | provenance support | panel b lower descendant | Keep as provenance support |
| `results/fig5_b3_line5_nosmooth_20260202_233434/Fig5_B3_LINE_*.pdf` | results PDFs | corrected no-smoothing band-wise diagnostics | provenance support | panel b lower descendant | Keep as provenance support |
| historical `Fig5_B3_POLAR_ESTIMATION` | legacy panel asset | old polar summary | legacy | panel b lower predecessor | Legacy only |

Interpretation:

- The original polar subpanel is no longer active.
- Its manuscript role has been replaced by a richer, code-backed band-decomposition branch.
- The band-wise PDFs are not side notes anymore; they are now active manuscript descendants.

---

## 3. Current figure families and whether they belong to the lineage

### Assets that are in the Master Figure 3 lineage

| Asset family | Why it belongs |
|---|---|
| `paper/figures/fig05_structure-macro-selection.png` | current active descendant of old `a + c` |
| `paper/figures/fig06_angle-specific-mechanism.png` | current active descendant of old `b` upper |
| `paper/figures/fig07_bandwise-routing-analysis-part1.png` | current active descendant of the band-wise replacement for old `b` lower |
| `paper/figures/fig08_bandwise-routing-analysis-part2.png` | current active descendant of the same band-wise branch |
| `figures/output/fig05_structure_alignment.pdf` | upstream generator split for old `a + c` |
| `figures/output/fig09_angle55.pdf`, `figures/output/fig09_angle100.pdf` | upstream generator assets for the active angle-specific mechanism figure |
| `results/fig5_b3_line5_*/*.pdf` | committed results support for the band-wise mechanism branch |

### Assets commonly confused with the lineage, but actually separate

| Asset family | Visual content summary | Classification | Why it is separate |
|---|---|---|---|
| `paper/figures/fig09_cross-material-universality.jpg` | cross-material objects + per-material heatmaps + RMSE statistics | separate family, not in Master Figure 3 lineage | this is the old `Master Figure 4` / cross-material branch |
| `paper/figures/fig09_heatmaps.pdf` | confusion-matrix heatmaps | separate supplementary family | this is a final-outcome ablation diagnostic, not a selection-probability mechanism panel |
| `paper/figures/fig04_unrolled-attention-omp.jpg` | architecture diagram | separate family | architecture figure |
| `paper/figures/fig04_noise-robustness-ablation.jpg` | robustness + ablation figure | separate family | performance/ablation figure |

---

## 4. Numbering trap that caused the confusion

### Stage A: original combined mechanism figure

| Label in that era | Actual content |
|---|---|
| `Master Figure 3` | one combined figure containing the global structure panel, the micro-level case-study panel, and the macro-robustness panel |
| `Master Figure 4` | separate cross-material family |

### Stage B: split-era naming

| Label in split era | Actual content |
|---|---|
| `Fig. 5 Structure Alignment` | only the old `a + c` portions of Master Figure 3 |
| `Fig. 6 Routing Mechanism` | only the old `b` portion of Master Figure 3 |
| `Fig. 7 Cross-material` | old `Master Figure 4` family |
| `Supp Fig. 9` | confusion family, previously `Supp Fig. 8` |

### Stage C: current manuscript numbering

| Current label | Actual content |
|---|---|
| `Fig. 5` | restored direct descendant of old `a + c` |
| `Fig. 6` | active angle-specific descendant of old `b` upper |
| `Fig. 7` | active band-wise descendant of old `b` lower (part I) |
| `Fig. 8` | active band-wise descendant of old `b` lower (part II) |
| `Fig. 9` | cross-material family from old Master Figure 4 |
| `Supp Fig. 9` | confusion family |

---

## 5. Decision table

### What is the canonical active descendant of each original block?

| Original source | Current canonical active descendant | Other descendants / support assets | Recommended keep surface |
|---|---|---|---|
| `Master Figure 3a + 3c` | `paper/figures/fig05_structure-macro-selection.png` | `fig05_structure_alignment.pdf`, split panel PDFs, legacy `A*`/`C*` families | Keep current `Fig. 5` |
| `Master Figure 3b` upper | `paper/figures/fig06_angle-specific-mechanism.png` | `fig09_angle55.pdf`, `fig09_angle100.pdf`, legacy `B1*`/`B2*` families | Keep current `Fig. 6` |
| `Master Figure 3b` lower replacement branch | `paper/figures/fig07_bandwise-routing-analysis-part1.png` and `paper/figures/fig08_bandwise-routing-analysis-part2.png` | `Fig5_B3_LINE_*` result PDFs, old `Fig5_B3_POLAR_ESTIMATION` | Keep current `Figs. 7–8`; keep polar only as legacy provenance |

---

## 6. Practical conclusion

If the question is only “which current things really came out of the original Master Figure 3?”, the answer is:

1. The current active descendant family is no longer a single `Fig. 5`; it is `Figs. 5–8`.
2. `Fig. 5` restores the original global-structure plus macro-robustness lineage.
3. `Fig. 6` replaces the old upper micro-mechanism block with cleaner angle-specific distributions.
4. `Figs. 7–8` elevate the old band-wise support PDFs into active manuscript figures.
5. The old polar block is no longer part of the active paper.
6. Current `Fig. 9` and `Supp Fig. 9` are separate families and should not be counted as Master Figure 3 descendants.
