# Figure Layout Specification

Single source of truth for intended panel-mm layout parameters for the active six main-paper figures.
Generator hardcoded values **must** match this document.
Use `paper/figures/*.layout.json` as realized-geometry checks, not as the design authority.

---

## Nature Communications Constraints

| Item               | Value          |
|--------------------|----------------|
| Single column      | 89 mm          |
| Double column      | 183 mm         |
| Max figure height  | **170 mm**     |
| Page size          | 210 × 276 mm  |
| Panel label        | 8pt bold lower |
| Figure text        | 5–7pt          |
| Font               | Arial/Helvetica|
| Min line width     | 0.5pt          |

Source: `docs/nature-communications/nature-communications-submission-requirements.md`

---

## Runtime Layout Contract

<!-- runtime-layout-contract:start -->
```yaml
version: 1
source_layout_spec: figures/conf/layout_spec.md
fonts:
  panel_label: 8.0
  title: 7.0
  axis_label: 6.5
  tick_label: 6.0
  legend: 6.5
  annotation: 6.5
  colorbar_tick: 5.5
  colorbar_label: 5.5
semantic_palette:
  physics: "#0072B2"
  learned: "#009E73"
  ablation: "#D55E00"
  highlight: "#E69F00"
  classical: "#E69F00"
stroke_tokens:
  base: 0.9
  data: 1.0
  emphasis: 1.1
  grid: 0.5
  annotation: 0.7
family_style:
  compact_marker_pt: 2.5
  standard_marker_pt: 3.0
  compact_capsize_pt: 1.8
  fill_alpha_primary: 0.12
  fill_alpha_secondary: 0.10
  fill_alpha_tertiary: 0.08
  summary_stem_alpha: 0.30
  summary_point_alpha: 0.68
  grid_alpha: 0.28
style_colors:
  neutral_text: "#1A1A1A"
  muted_text: "#666666"
  dense_routing: "#4A4A4A"
  guide_line: "#8F8F8F"
  guide_fill: "#E7E7E7"
  chance_fill: "#F5F5F5"
  chance_line: "#CFCFCF"
  highlight_fill: "#F7E8AD"
figures:
  fig01:
    compose:
      width_mm: 183.0
      height_mm: 144.0
      row_height_mm: 60.0
      bottom_row_height_mm: 80.0
      row_gap_mm: 4.0
      panel_a_width_mm: 54.0
      panel_b_width_mm: 124.0
    generator:
      composite_width_mm: 183.0
      composite_height_mm: 80.0
      composite_grid:
        left: 0.055
        right: 0.985
        bottom: 0.12
        top: 0.92
        wspace: 0.28
        width_ratios: [1.0, 1.0, 1.0]
  fig02:
    compose:
      width_mm: 183.0
      height_mm: 130.0
    generator:
      composite_width_mm: 183.0
      composite_height_mm: 130.0
      composite_grid:
        left: 0.07
        right: 0.96
        bottom: 0.07
        top: 0.95
        hspace: 0.35
        wspace: 0.40
        width_ratios: [1.0, 1.0, 1.0]
        height_ratios: [1.0, 1.0]
  fig03:
    compose:
      width_mm: 183.0
      height_mm: 130.0
    generator:
      composite_width_mm: 183.0
      composite_height_mm: 130.0
      composite_grid:
        left: 0.07
        right: 0.96
        bottom: 0.07
        top: 0.95
        hspace: 0.35
        wspace: 0.40
        width_ratios: [1.0, 1.0, 1.0]
        height_ratios: [1.0, 1.0]
  fig04:
    compose:
      width_mm: 183.0
      height_mm: 140.0
      outer_margin_x_mm: 4.0
      outer_margin_y_mm: 4.0
    generator:
      composite_width_mm: 183.0
      composite_height_mm: 140.0
      composite_grid:
        left: 0.055
        right: 0.985
        bottom: 0.085
        top: 0.965
        hspace: 0.34
        wspace: 0.28
        width_ratios: [1.0, 1.0, 1.0]
        height_ratios: [1.0, 1.0]
      split:
        panel_slot_width_mm: {a: 56.0, b: 56.0, c: 56.0, d: 56.0, e: 56.0, f: 56.0}
        panel_slot_height_mm: {a: 54.0, b: 54.0, c: 54.0, d: 54.0, e: 54.0, f: 54.0}
        standalone_subplots:
          a: {left: 0.130, right: 0.970, bottom: 0.180, top: 0.900}
          b: {left: 0.130, right: 0.970, bottom: 0.180, top: 0.900}
          c: {left: 0.130, right: 0.970, bottom: 0.180, top: 0.900}
          d: {left: 0.130, right: 0.970, bottom: 0.180, top: 0.900}
          e: {left: 0.130, right: 0.970, bottom: 0.180, top: 0.900}
          f: {left: 0.130, right: 0.970, bottom: 0.180, top: 0.900}
  fig05:
    compose:
      width_mm: 183.0
      height_mm: 188.0
    generator:
      composite_width_mm: 183.0
      composite_height_mm: 188.0
      composite_grid:
        left: 0.055
        right: 0.985
        bottom: 0.055
        top: 0.975
        hspace: 0.330
        height_ratios: [46.0, 58.0, 42.0]
      top_row:
        wspace: 0.320
        width_ratios: [1.0, 1.0, 0.9]
      middle_row:
        wspace: 0.220
        width_ratios: [1.42, 0.92]
      bottom_row:
        wspace: 0.220
        width_ratios: [1.0, 1.0]
      confusion_pair: {wspace: 0.10, colorbar_ratio: 0.065}
      standalone:
        a:
          width_mm: 183.0
          height_mm: 68.0
          subplots_adjust: {left: 0.10, right: 0.95, bottom: 0.15, top: 0.92}
        b:
          width_mm: 183.0
          height_mm: 68.0
          subplots_adjust: {left: 0.10, right: 0.95, bottom: 0.15, top: 0.92}
        c:
          width_mm: 183.0
          height_mm: 78.0
          subplots_adjust: {left: 0.11, right: 0.96, bottom: 0.14, top: 0.90}
        d:
          width_mm: 183.0
          height_mm: 112.0
          grid: {left: 0.07, right: 0.965, bottom: 0.10, top: 0.93, wspace: 0.10, colorbar_ratio: 0.040}
        e:
          width_mm: 183.0
          height_mm: 78.0
          subplots_adjust: {left: 0.11, right: 0.96, bottom: 0.13, top: 0.90}
        f:
          width_mm: 183.0
          height_mm: 78.0
          subplots_adjust: {left: 0.12, right: 0.96, bottom: 0.20, top: 0.88}
        g:
          width_mm: 183.0
          height_mm: 72.0
          subplots_adjust: {left: 0.10, right: 0.95, bottom: 0.16, top: 0.88}
  fig06:
    compose:
      width_mm: 183.0
      height_mm: 170.0
      outer_margin_x_mm: 6.0
      outer_margin_top_mm: 3.0
      outer_margin_bottom_mm: 3.0
      panel_a_height_mm: 29.0
      panel_b_height_mm: 24.0
      row_cd_height_mm: 47.0
      panel_d_width_mm: 83.0
      panel_e_height_mm: 55.0
      col_gap_mm: 4.0
      row_gap_mm: 3.0
    typography:
      panel_label: 8.0
      title: 6.8
      axis_label: 6.3
      tick_label: 5.8
      legend: 6.0
      colorbar_tick: 5.5
      colorbar_label: 5.5
    generator:
      composite_width_mm: 183.0
      composite_height_mm: 170.0
      composite_grid:
        left: 0.055
        right: 0.945
        bottom: 0.065
        top: 0.955
        hspace: 0.34
        wspace: 0.0
        width_ratios: [1.0]
        height_ratios: [29.0, 24.0, 47.0, 55.0]
      middle_row:
        wspace: 0.18
        width_ratios: [84.0, 83.0]
      split:
        standalone:
          a:
            width_mm: 171.0
            height_mm: 29.0
            subplots_adjust: {left: 0.08, right: 0.985, bottom: 0.26, top: 0.88}
          b:
            width_mm: 171.0
            height_mm: 24.0
            subplots_adjust: {left: 0.06, right: 0.985, bottom: 0.18, top: 0.92}
          c:
            width_mm: 84.0
            height_mm: 47.0
            subplots_adjust: {left: 0.11, right: 0.985, bottom: 0.17, top: 0.92}
          d:
            width_mm: 83.0
            height_mm: 47.0
            subplots_adjust: {left: 0.12, right: 0.985, bottom: 0.18, top: 0.92}
          e:
            width_mm: 171.0
            height_mm: 55.0
            subplots_adjust: {left: 0.08, right: 0.985, bottom: 0.13, top: 0.94}
```
<!-- runtime-layout-contract:end -->

---

## Unified Font Specification

```yaml
fonts:
  panel_label:    8.0pt bold    # a, b, c, ...
  title:          7.0pt         # panel title
  axis_label:     6.5pt         # xlabel, ylabel
  tick_label:     6.0pt         # xtick, ytick
  legend:         6.5pt         # main legend text
  annotation:     6.5pt         # stat text, d=1.95
  colorbar_tick:  5.5pt         # colorbar numbers
  colorbar_label: 5.5pt         # only if space permits
```

All font sizes stay within Nature's `5–7 pt` non-panel band, but the branch default should not sit on the lower edge unless an explicit exception is justified here.

---

## Matplotlib Spacing Formulas

```
usable_width  = width_mm  × (right − left)
usable_height = height_mm × (top − bottom)

col_width  = usable_width  / (N_cols + (N_cols−1) × wspace)
row_height = usable_height / (N_rows + (N_rows−1) × hspace)

panel_width  = n_span × col_width + (n_span−1) × wspace × col_width
```

---

## Figure 1 — Direction-Dependent Structural Filtering (5 panels)

Composed: external a,b (top) + generated c,d,e (bottom strip)

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌─────────────────────────┬───────────────────────────┐ ─┬─
│                         │                           │  │
│   (a) Setup photo       │   (b) Shared-response      │  │ 65.0 mm
│   68.0 × 65.0 mm       │   reweighting view         │  │ (external)
│                         │   110.0 × 65.0 mm         │  │
│                         │                           │  │
├─────────────────────────┴───────────────────────────┤ ─┼─  ← 5mm compose gap
│                                                     │  │
│  Generated strip: 183.0 × 65.0 mm                   │  │
│  GridSpec 1×3, wspace=0.40                           │  │
│  Margins: L=0.06 R=0.97 B=0.18 T=0.88               │  │
│  Usable: 166.5 × 45.5 mm                            │  │
│                                                     │  │ 65.0 mm
│  ┌──────────────┬──────────────┬──────────────┐      │  │ (generated)
│  │(c) Input→Out │(d) Angle-freq│(e) Directiv. │      │  │
│  │ 5 angles     │ heatmap      │ polar, 4band │      │  │
│  │ 43.8 × 45.5  │ 43.8 × 45.5  │ 43.8 × 45.5 │      │  │
│  └──────────────┴──────────────┴──────────────┘      │  │
│  ├── 43.8mm ──┤wsp├── 43.8mm ──┤wsp├── 43.8mm ──┤   │  │
│               17.5              17.5                  │  │
└─────────────────────────────────────────────────────┘ ─┴─
                                          total ≈ 135.0 mm
```

| Parameter   | Value                              |
|-------------|------------------------------------|
| width_mm    | 183                                |
| height_mm   | 65                                 |
| GridSpec    | 1×3, wspace=0.40                   |
| Margins     | L=0.06, R=0.97, B=0.18, T=0.88    |
| Usable      | 166.5 × 45.5 mm                    |
| col_width   | 166.5 / (3 + 2×0.40) = **43.8mm** |

---

## Figure 2 — SVD Spectrum (6 panels)

Full generated figure (no external assets).

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────────────────────────────┐ ─┬─
│  GridSpec 2×3, hspace=0.35, wspace=0.40              │  │
│  Margins: L=0.07 R=0.96 B=0.07 T=0.95               │  │
│  Usable: 162.9 × 114.4 mm                           │  │
│  col_width = 162.9 / (3 + 2×0.40) = 42.9 mm         │  │
│  row_height = 114.4 / (2 + 0.35) = 48.7 mm          │  │
│                                                      │  │
│  Row 1 — 48.7 mm                                     │  │
│  ┌──────────────┬──────────────┬──────────────┐      │  │
│  │(a) SVD       │(b) Mode freq │(c) Mode polar│      │  │
│  │spectrum      │ profiles     │ patterns     │      │  │
│  │42.9 × 48.7   │42.9 × 48.7   │42.9 × 48.7  │      │  │
│  └──────────────┴──────────────┴──────────────┘      │  │ 130.0 mm
│  ├── 42.9 ──┤17.2├── 42.9 ──┤17.2├── 42.9 ──┤       │  │
│                                                      │  │
│        inter-row gap = 0.35 × 48.7 = 17.0 mm        │  │
│                                                      │  │
│  Row 2 — 48.7 mm                                     │  │
│  ┌──────────────┬──────────────┬──────────────┐      │  │
│  │(d) H heatmap │(e) Reconstr. │(f) Corr mat  │      │  │
│  │42.9 × 48.7   │42.9 × 48.7   │42.9 × 48.7  │      │  │
│  └──────────────┴──────────────┴──────────────┘      │  │
└──────────────────────────────────────────────────────┘ ─┴─
```

| Parameter       | Value                                |
|-----------------|--------------------------------------|
| width_mm        | 183                                  |
| height_mm       | **130** (was 170, saved 40mm)        |
| GridSpec        | 2×3, hspace=0.35, wspace=0.40       |
| Margins         | L=0.07, R=0.96, B=0.07, T=0.95      |
| Usable          | 162.9 × 114.4 mm                     |
| col_width       | **42.9 mm**                          |
| row_height      | **48.7 mm**                          |
| Inter-row gap   | 0.35 × 48.7 = **17.0 mm**           |
| wspace gap      | 0.40 × 42.9 = **17.2 mm**           |

---

## Figure 3 — WN vs Speech Discriminability (6 panels)

Full generated figure. Compares white noise and speech encoding/decoding.

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────────────────────────────┐ ─┬─
│  GridSpec 2×3, hspace=0.35, wspace=0.40              │  │
│  Margins: L=0.07 R=0.96 B=0.07 T=0.95               │  │
│  Usable: 162.9 × 114.4 mm                           │  │
│  col_width = 162.9 / (3 + 2×0.40) = 42.9 mm         │  │
│  row_height = 114.4 / (2 + 0.35) = 48.7 mm          │  │
│                                                      │  │
│  Row 1 — 48.7 mm                                     │  │
│  ┌──────────────┬──────────────┬──────────────┐      │  │
│  │(a) WN violin │(b) Sp violin │(c) Discrim.  │      │  │
│  │ within/betw  │ within/betw  │ margin W/S   │      │  │
│  │42.9 × 48.7   │42.9 × 48.7   │42.9 × 48.7  │      │  │
│  └──────────────┴──────────────┴──────────────┘      │  │
│  ├── 42.9 ──┤17.2├── 42.9 ──┤17.2├── 42.9 ──┤       │  │ 130.0 mm
│                                                      │  │
│        inter-row gap = 0.35 × 48.7 = 17.0 mm        │  │
│                                                      │  │
│  Row 2 — 48.7 mm                                     │  │
│  ┌──────────────┬──────────────┬──────────────┐      │  │
│  │(d) OMP acc   │(e) Split sim │(f) Dose-resp │      │  │
│  │ WN vs Speech │ WN / Speech  │ SNR sweep    │      │  │
│  │42.9 × 48.7   │42.9 × 48.7   │42.9 × 48.7  │      │  │
│  └──────────────┴──────────────┴──────────────┘      │  │
└──────────────────────────────────────────────────────┘ ─┴─
```

| Parameter       | Value                                |
|-----------------|--------------------------------------|
| width_mm        | 183                                  |
| height_mm       | **130**                              |
| GridSpec        | 2×3, hspace=0.35, wspace=0.40       |
| Margins         | L=0.07, R=0.96, B=0.07, T=0.95      |
| Usable          | 162.9 × 114.4 mm                     |
| col_width       | **42.9 mm**                          |
| row_height      | **48.7 mm**                          |
| Inter-row gap   | 0.35 × 48.7 = **17.0 mm**           |
| wspace gap      | 0.40 × 42.9 = **17.2 mm**           |

---

## Figure 4 — Solver Dynamics (4 panels)

Composed: a full-width admissibility thesis strip `a`, followed by three
equal-weight lower panels `b/c/d`. The figure should read as
measured neighborhood -> representative broad-to-local transition ->
validation-wide contraction -> angle-resolved within-15° contraction. The design is
population-led, but it keeps one light exemplar so the broad-to-local shape
remains visible before the chapter hands its radius-based logic to Fig. 5.

```
                         183.0 mm
┌──────────────────────────────────────────────────────┐
│                4.0 mm manuscript collar              │
│  ┌────────────────────────────────────────────────┐  │
│  │ (a) Admissibility strip                        │  │ 36.0 mm slot
│  │              175.0 × 36.0 mm                  │  │
│  └────────────────────────────────────────────────┘  │
│                    4.0 mm row gap                    │
│  ┌────────────────┬────────────────┬─────────────┐  │
│  │ (b) Exemplary  │ (c) Validation │ (d) Angle × │  │
│  │ broad-to-local │ wide           │ radius      │  │ 52.0 mm
│  │ transition     │ contraction    │ map         │  │
│  │ 55.7 × 52.0 mm │ 55.7 × 52.0 mm │ 55.7 × 52.0 │  │
│  └────────────────┴────────────────┴─────────────┘  │
│                4.0 mm manuscript collar              │
└──────────────────────────────────────────────────────┘
                              total = 132.0 mm
```

| Parameter      | Value                                  |
|----------------|----------------------------------------|
| width_mm       | 183                                    |
| height_mm      | **132**                                |
| Outer collar   | **4.0 mm** on all four sides           |
| Panel (a) slot | **175.0 × 36.0 mm**                    |
| Row gap        | **4.0 mm**                             |
| Panels (b-d)   | **55.7 × 52.0 mm**                     |
| Column gap     | **4.0 mm** between lower panels        |
| Label lane     | **4.5 mm** at the top of each slot     |
| Content inset  | **1.5 mm** left/right/bottom           |

---

## Figure 5 — Performance + Structure (5 panels)

Full generated figure rebuilt as a compact two-row journal layout with
decorated-bbox budgeting, not raw-slot-only packing:

- Top row `a-b-c`: benchmark + unified confusion family + structure anchor
- Bottom row `d-e`: four-angle conditional outputs + per-angle benchmark

Panel `b` remains the dominant confusion-family block, but `c` is widened so
the physical-manifold anchor does not read like a side note. `c` keeps its own
composite colorbar and shared axis labels. The larger inter-row gap is
intentional and exists to prevent decorated text from colliding across rows.

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────────────────────────────┐ ─┬─
│  Two-row layout                                       │  │
│  Margins: L=7 R=7 B=7 T=7 mm                         │  │
│  Usable: 169 × 142 mm                               │  │
│  Row heights: 62 mm / 64 mm                         │  │
│  Row gap: 16 mm                                     │  │
│                                                      │  │
│  Top row                                             │  │
│  ┌──────────────┬──────────────────────┬─────────┐   │  │
│  │ (a)          │ (b) unified confusion│ (c)     │   │  │
│  │ benchmark    │ family block         │ manifold│   │  │ 62 mm
│  │ 46 × 62 mm   │ 64 × 62 mm           │ 51×62   │   │  │
│  └──────────────┴──────────────────────┴─────────┘   │  │
│      4 mm gap              4 mm gap                  │  │
│                                                      │  │
│  Bottom row                                          │  │
│  ┌─────────────────────────────┬─────────────────┐   │  │
│  │ (d) conditional outputs     │ (e) per-angle   │   │  │
│  │ 2x2 angles: 55 / 70 / 95 /  │ benchmark       │   │  │ 64 mm
│  │ 100, 90 × 64 mm             │ 72 × 64 mm      │   │  │
│  └─────────────────────────────┴─────────────────┘   │  │
│                 7 mm gap                             │  │
└──────────────────────────────────────────────────────┘ ─┴─
                                          total = 156.0 mm
```

| Parameter             | Value                                      |
|-----------------------|--------------------------------------------|
| width_mm              | 183                                        |
| height_mm             | **156**                                    |
| Outer rows            | 2×1, height ratios 62 / 64                 |
| Top row               | 1×3, width ratios 46 / 64 / 51             |
| Bottom row            | 1×2, width ratios 90 / 72                  |
| Margins               | 7 mm on all sides                          |
| Row gap               | 16 mm                                      |
| Panel (a)             | 46 × 62 mm                                 |
| Panel (b)             | 64 × 62 mm                                 |
| Panel (c)             | 51 × 62 mm                                 |
| Panel (d)             | 90 × 64 mm                                 |
| Panel (e)             | 72 × 64 mm                                 |

---

## Figure 6 — Universality (5 panels)

Composed: a full-width measured response-regime map `a`, a full-width generated `H` row `b`, then a shared row with local-ordering decay `c` on the left and readout-versus-overlap summary `d` on the right, followed by a full-width object-conditioned contrast-band row `e`.
Panel `a` is rebuilt through the governed generator path rather than passed through a cropped legacy support strip.
Panels `a-e` are generated at their final manuscript slot sizes.

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────────────────────────────┐ ─┬─
│                outer top margin 4 mm                │  │
├──────────────────────────────────────────────────────┤ ─┼─
│   (a) 171.0 × 28.0 mm response-regime map           │  │ 28.0 mm
├──────────────────────────────────────────────────────┤ ─┼─ 4.0 mm
│      (b) 171.0 × 29.0 mm cross-material H row       │  │ 29.0 mm
├───────────────────────────────┬──────────────────────┤ ─┼─ 4.0 mm
│ (c) 84.0 × 48.0 mm            │ (d) 83.0 × 48.0 mm  │  │ 48.0 mm
│ Local-ordering decay          │ Readout summary      │  │
├──────────────────────────────────────────────────────┤ ─┼─ 4.0 mm
│ (e) 171.0 × 43.0 mm contrast-band + code row        │  │ 43.0 mm
└───────────────────────────────┴──────────────────────┘ ─┼─
                                                        ─┴─ 4.0 mm bottom margin
```

| Parameter         | Value                       |
|------------------|-----------------------------|
| width_mm         | 183                         |
| height_mm        | **165**                     |
| outer_margin_x   | **6.0 mm** each side        |
| outer_margin_top | **4.0 mm**                  |
| outer_margin_bot | **4.0 mm**                  |
| Panel (a)        | **171.0 × 28.0 mm**         |
| Panel (b)        | **171.0 × 29.0 mm**         |
| Panel (c)        | **84.0 × 48.0 mm**          |
| Panel (d)        | **83.0 × 48.0 mm**          |
| Panel (e)        | **171.0 × 43.0 mm**         |
| row_gap          | **4.0 mm**                  |
| col_gap          | **4.0 mm**                  |
