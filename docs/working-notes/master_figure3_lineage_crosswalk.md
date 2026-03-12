# Master Figure 3 Lineage Crosswalk

Date: 2026-03-12

Purpose: disentangle which current figures, generator PDFs, results PDFs, and legacy assets actually descend from the original `Master Figure 3` and which only look nearby because of later numbering changes (`Fig. 5/6/7/8/9`, supplementary renumbering, and split-era generator outputs).

This note is a working crosswalk. Canonical provenance remains in:

- `figures/FIGURE_REGISTRY.md`
- `figures/conf/review_targets.yaml`
- `figures/conf/experiments.yaml`

## Reading rule

Each asset in this note is classified as one of:

- `panel a descendant`
- `panel b descendant`
- `panel c descendant`
- `combined a+b+c descendant`
- `partial derivative of a+c`
- `panel-b support variant`
- `separate family, not in Master Figure 3 lineage`

The classification uses visual content first, then file naming, registry metadata, and git history as provenance support.

---

## 1. Original anchor: what Master Figure 3 actually is

### Canonical original asset

| Asset | Type | Visual title / content | Classification |
|---|---|---|---|
| `old/Figure-5.png` | legacy image | Visually shows `Master Figure 3 (Final): Deciphering AI Understanding via Micro-Mechanism & Macro-Robustness` with three panels: global attention, micro-level case study, macro-level selection robustness | Original anchor |

### Original panel semantics

| Original panel | Visual content | Scientific message |
|---|---|---|
| `a` | diagonal heatmap titled `Global Physics-Consistent Attention` | the model learns a global routing structure aligned with the physical angle manifold |
| `b` | `Micro-level Mechanism` case study with bar/stem plots and a polar estimate | on one representative sample, analytical OMP makes a spurious off-axis choice while the physics-aware model stays sparse and aligned with the true DOA |
| `c` | two all-angle selection probability heatmaps | globally, OMP spreads mass off diagonal while the physics-aware model remains sharply diagonal |

---

## 2. Direct descendants by original panel

### Panel a descendants

| Asset | Type | Visual content summary | Current status | Classification | Recommendation |
|---|---|---|---|---|---|
| `paper/figures/fig05_routing-mechanism-analysis.png` (panel a region) | current manuscript composite | same global diagonal-attention panel still appears in the current manuscript composite | current main | panel a descendant | Keep as canonical active descendant |
| `figures/output/fig05_routing_mechanism_analysis_panels/fig05_panel_a_global_attention.pdf` | split panel PDF | isolated current panel-a version of the global structure panel | current panel asset | panel a descendant | Keep as canonical isolated panel |
| `paper/figures/fig05_structure_alignment.pdf` | current generator PDF | left half contains the global structure pair (`H` structure + `QK` structure) that replaces the old single visual panel-a view with a cleaner split presentation | current generator output | partial derivative of a+c | Keep as active upstream support |
| `figures/output/fig05_structure_alignment.pdf` | current generator PDF | same content as the paper copy | current generator output | partial derivative of a+c | Support only |
| historical `Fig5_A1_PHYSICAL_STRUCTURE` | legacy panel asset family | physical-structure side of the old panel-a decomposition | legacy | panel a descendant | Legacy only |
| historical `Fig5_A2_QK_STRUCTURE` | legacy panel asset family | learned-structure side of the old panel-a decomposition | legacy | panel a descendant | Legacy only |

Interpretation:

- The old panel `a` did not become a separate new manuscript figure.
- It was decomposed into cleaner structure-alignment assets, then recomposed into current `Fig. 5a`.

### Panel b descendants

| Asset | Type | Visual content summary | Current status | Classification | Recommendation |
|---|---|---|---|---|---|
| `paper/figures/fig05_routing-mechanism-analysis.png` (panel b region) | current manuscript composite | visually still shows the old micro-level mechanism layout with the bar/stem case study and polar estimate | current main | panel b descendant | Keep as current canonical composite descendant |
| `paper/figures/fig06_routing_mechanism.pdf` | current generator PDF | updated micro-level mechanism view: 2x2 OMP/QK heatmaps plus five band-wise line plots | current generator output | panel b descendant | Keep as active upstream support |
| `figures/output/fig06_routing_mechanism.pdf` | current generator PDF | same as paper copy | current generator output | panel b descendant | Support only |
| `results/fig5_b3_line5_20260202_224349/Fig5_B3_LINE_300_3000.pdf` and sibling band PDFs | results PDFs | band-wise line-plot support assets for the panel-b case-study logic | results support | panel-b support variant | Legacy support only |
| `results/fig5_b3_line5_nosmooth_20260202_233434/Fig5_B3_LINE_300_3000.pdf` and sibling band PDFs | results PDFs | no-smoothing corrected band-wise support assets for the same case-study lineage | results support | panel-b support variant | Legacy support only |
| historical `Fig5_B1_*` | legacy panel asset family | early physics case-study panels | legacy | panel b descendant | Legacy only |
| historical `Fig5_B2_*` | legacy panel asset family | early QK case-study panels | legacy | panel b descendant | Legacy only |
| historical `Fig5_B3_POLAR_ESTIMATION` | legacy panel asset family | early polar estimate asset from the same micro-level mechanism lineage | legacy | panel b descendant | Legacy only |

Interpretation:

- Panel `b` is the most confusing branch because the current paper-facing composite still visually resembles the original `Master Figure 3b`, while the upstream generator PDF has already evolved to `heatmaps + band lines`.
- The `Fig5_B3_LINE_*` PDFs are not a separate figure family. They are support descendants of the same panel-b mechanism branch.

### Panel c descendants

| Asset | Type | Visual content summary | Current status | Classification | Recommendation |
|---|---|---|---|---|---|
| `paper/figures/fig05_routing-mechanism-analysis.png` (panel c region) | current manuscript composite | same macro-level twin heatmap concept still appears in the current manuscript composite | current main | panel c descendant | Keep as canonical active descendant |
| `figures/output/fig05_routing_mechanism_analysis_panels/fig05_panel_c_macro_robustness.pdf` | split panel PDF | isolated current panel-c version | current panel asset | panel c descendant | Keep as canonical isolated panel |
| `paper/figures/fig05_structure_alignment.pdf` | current generator PDF | right half contains the macro selection-probability heatmaps | current generator output | partial derivative of a+c | Keep as active upstream support |
| `figures/output/fig05_structure_alignment.pdf` | current generator PDF | same content as the paper copy | current generator output | partial derivative of a+c | Support only |
| historical `Fig5_C1_PHYSICS_SELECTION` | legacy panel asset family | old OMP macro-probability panel | legacy | panel c descendant | Legacy only |
| historical `Fig5_C2_QK_SELECTION` | legacy panel asset family | old QK macro-probability panel | legacy | panel c descendant | Legacy only |

Interpretation:

- The old panel `c` stayed conceptually stable.
- Like panel `a`, it was peeled into cleaner generator-backed assets and then recomposed into current `Fig. 5c`.

---

## 3. Current figure families and whether they belong to the lineage

### Assets that are in the Master Figure 3 lineage

| Asset family | Why it belongs |
|---|---|
| `paper/figures/fig05_routing-mechanism-analysis.png` | current manuscript composite of the same structure + micro + macro story |
| `paper/figures/fig05_structure_alignment.pdf` / `figures/output/fig05_structure_alignment.pdf` | current upstream split of the old `a + c` content |
| `paper/figures/fig06_routing_mechanism.pdf` / `figures/output/fig06_routing_mechanism.pdf` | current upstream split of the old `b` content |
| `figures/output/fig05_routing_mechanism_analysis_panels/fig05_panel_a_global_attention.pdf` | isolated current descendant of panel `a` |
| `figures/output/fig05_routing_mechanism_analysis_panels/fig05_panel_c_macro_robustness.pdf` | isolated current descendant of panel `c` |
| `results/fig5_b3_line5_*/*.pdf` | support-only descendants of panel `b` |
| legacy `Fig5_A*`, `Fig5_B*`, `Fig5_C*` families | older atomic descendants of panels `a/b/c` |

### Assets commonly confused with the lineage, but actually separate

| Asset family | Visual content summary | Classification | Why it is separate |
|---|---|---|---|
| `paper/figures/fig06_cross-material-universality.jpg` | cross-material objects + per-material heatmaps + RMSE statistics | separate family, not in Master Figure 3 lineage | this is the old `Master Figure 4` / cross-material branch, not the structure-micro-macro mechanism branch |
| `old/Figure-6.jpg` | visually titled `Master Figure 4: Universal Physical Encoding Across Diverse Matter` | separate family, not in Master Figure 3 lineage | direct legacy predecessor of current cross-material figure family |
| `paper/figures/fig09_heatmaps.pdf`, `fig09_angle55.pdf`, `fig09_angle100.pdf` | confusion matrices and angle-specific supplementary diagnostics | separate family, not in Master Figure 3 lineage | this is the old supplementary confusion family, formerly `Supp Fig. 8`, later renumbered to `Supp Fig. 9` |
| removed `generate_figure8.py` / `results_figure8/*` | historical confusion-matrix generation path | separate family, not in Master Figure 3 lineage | old `Figure 8` content maps to the supplementary confusion branch, not to Master Figure 3 |
| `paper/figures/fig03_unrolled-attention-omp.jpg` | architecture diagram | separate family, not in Master Figure 3 lineage | architecture figure |
| `paper/figures/fig04_noise-robustness-ablation.jpg` | robustness + ablation figure | separate family, not in Master Figure 3 lineage | performance/ablation figure |

### Important absence: there is no active current Figure 7 or Figure 8

| Number | Actual status |
|---|---|
| `Fig. 7` | not an active final manuscript figure; during the split era it temporarily referred to the cross-material family, which later became current `Fig. 6` |
| `Fig. 8` | not an active final manuscript figure; the old `Figure 8` generation path belonged to the confusion-matrix supplementary family that is now `Supp Fig. 9` |

---

## 4. Numbering trap that caused the confusion

### Stage A: original combined mechanism figure

| Label in that era | Actual content |
|---|---|
| `Master Figure 3` | one combined figure containing the global structure panel, the micro-level case-study panel, and the macro robustness panel |
| `Master Figure 4` | separate cross-material family |

### Stage B: split-era naming

| Label in split era | Actual content |
|---|---|
| `Fig. 5 Structure Alignment` | only the old `a + c` portions of Master Figure 3 |
| `Fig. 6 Routing Mechanism` | only the old `b` portion of Master Figure 3 |
| `Fig. 7 Cross-material` | old `Master Figure 4` family |
| `Supp Fig. 9` | confusion family, previously `Supp Fig. 8` |

### Stage C: current manuscript final numbering

| Current label | Actual content |
|---|---|
| `Fig. 5` | recombined structure + micro + macro family descended from Master Figure 3 |
| `Fig. 6` | cross-material family descended from old Master Figure 4 |
| `Supp Fig. 9` | confusion family descended from old supplementary confusion assets |

---

## 5. Decision table

### What is the canonical active descendant of each original panel?

| Original source | Current canonical active descendant | Other descendants / support assets | Separate families often confused with it | Recommended keep surface |
|---|---|---|---|---|
| `Master Figure 3a` | `paper/figures/fig05_routing-mechanism-analysis.png` panel a | `fig05_panel_a_global_attention.pdf`, `fig05_structure_alignment.pdf`, historical `Fig5_A*` | none major beyond the general split-figure naming confusion | Keep the current Fig. 5 composite plus the panel-a split asset |
| `Master Figure 3b` | `paper/figures/fig05_routing-mechanism-analysis.png` panel b | `fig06_routing_mechanism.pdf`, `Fig5_B3_LINE_*.pdf`, historical `Fig5_B*` | cross-material and confusion PDFs are not descendants | Keep the current Fig. 5 composite; keep `fig06_routing_mechanism.pdf` as the active upstream support; keep `Fig5_B3_LINE_*` as provenance support only |
| `Master Figure 3c` | `paper/figures/fig05_routing-mechanism-analysis.png` panel c | `fig05_panel_c_macro_robustness.pdf`, `fig05_structure_alignment.pdf`, historical `Fig5_C*` | confusion PDFs can look heatmap-adjacent but are a different family | Keep the current Fig. 5 composite plus the panel-c split asset |

### Which current or legacy figure numbers are actually descendants?

| Figure label someone might mention | Does it belong to Master Figure 3 lineage? | Correct interpretation |
|---|---|---|
| current `Fig. 5` | yes | canonical recombined descendant |
| split-era `Fig. 5` | yes, partially | only the old `a + c` content |
| split-era `Fig. 6` | yes, partially | only the old `b` content |
| current `Fig. 6` | no | cross-material family from old Master Figure 4 |
| temporary `Fig. 7` | no | split-era name for cross-material |
| old `Figure 8` path | no | supplementary confusion family later renumbered |
| current `Supp Fig. 9` | no | confusion family, not a Master Figure 3 descendant |

---

## 6. Practical conclusion

If the question is only “which current things really came out of the original Master Figure 3?”, the answer is:

1. The true active descendant is the current `Fig. 5` family.
2. The current split generator PDFs divide that family into:
   - `fig05_structure_alignment.pdf` for old `a + c`
   - `fig06_routing_mechanism.pdf` for old `b`
3. The `Fig5_B3_LINE_*` PDFs are not a new figure family; they are panel-b support descendants.
4. Current `Fig. 6`, old temporary `Fig. 7`, old `Figure 8`, and current `Supp Fig. 9` are separate families and should not be treated as descendants of `Master Figure 3`.
