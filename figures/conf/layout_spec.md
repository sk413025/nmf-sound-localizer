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
figures:
  fig04:
    compose:
      width_mm: 183.0
      height_mm: 154.0
      outer_margin_mm: 4.0
      panel_a_slot_width_mm: 175.0
      panel_a_slot_height_mm: 74.0
      row_gap_mm: 4.0
      col_gap_mm: 4.0
      lower_panel_slot_width_mm: 55.667
      lower_panel_slot_height_mm: 68.0
      panel_d_slot_height_mm: 43.0
      panel_e_slot_height_mm: 22.0
      right_stack_gap_mm: 3.0
      label_lane_mm: 4.5
      content_inset_x_mm: 1.5
      content_inset_bottom_mm: 1.5
    generator:
      composite_width_mm: 183.0
      composite_height_mm: 76.0
      composite_grid:
        left: 0.075
        right: 0.990
        bottom: 0.08
        top: 0.98
        wspace: 0.28
        width_ratios: [58.333, 58.333, 58.333]
        stack_height_ratios: [43.0, 22.0]
        stack_hspace: 0.34
      split:
        panel_slot_width_mm: {b: 58.333, c: 58.333, d: 58.333, e: 58.333}
        panel_slot_height_mm: {b: 76.0, c: 76.0, d: 45.0, e: 23.0}
        standalone_subplots:
          b: {left: 0.14, right: 0.995, bottom: 0.08, top: 0.980}
          c: {left: 0.12, right: 0.995, bottom: 0.08, top: 0.980}
          d: {left: 0.14, right: 0.995, bottom: 0.12, top: 0.985}
          e: {left: 0.12, right: 0.995, bottom: 0.28, top: 0.94}
  fig05:
    generator:
      composite_width_mm: 183.0
      composite_height_mm: 156.0
      composite_grid:
        left: 0.038
        right: 0.962
        bottom: 0.045
        top: 0.955
        hspace: 0.254
        height_ratios: [62.0, 64.0]
      top_row:
        wspace: 0.110
        width_ratios: [48.0, 73.0, 40.0]
      bottom_row:
        wspace: 0.086
        width_ratios: [90.0, 72.0]
      heatmap_stack: {hspace: 0.50, wspace: 0.06, colorbar_ratio: 0.08}
      routing_stack: {hspace: 0.42, wspace: 0.18}
      standalone:
        a:
          width_mm: 183.0
          height_mm: 70.0
          subplots_adjust: {left: 0.10, right: 0.95, bottom: 0.15, top: 0.92}
        b:
          width_mm: 183.0
          height_mm: 120.0
          grid: {left: 0.08, right: 0.95, bottom: 0.10, top: 0.93, hspace: 0.30, wspace: 0.08, colorbar_ratio: 0.06}
        c:
          width_mm: 183.0
          height_mm: 120.0
          grid: {left: 0.08, right: 0.95, bottom: 0.08, top: 0.94, hspace: 0.30, wspace: 0.06, colorbar_ratio: 0.08}
        d:
          width_mm: 183.0
          height_mm: 118.0
          grid: {left: 0.08, right: 0.96, bottom: 0.12, top: 0.91, hspace: 0.40, wspace: 0.18}
        e:
          width_mm: 183.0
          height_mm: 70.0
          subplots_adjust: {left: 0.08, right: 0.95, bottom: 0.15, top: 0.92}
  fig06:
    compose:
      width_mm: 183.0
      height_mm: 202.0
      outer_margin_x_mm: 6.0
      outer_margin_top_mm: 4.0
      outer_margin_bottom_mm: 4.0
      panel_a_height_mm: 32.0
      panel_b_height_mm: 32.0
      panel_c_height_mm: 56.0
      panel_d_height_mm: 62.0
      panel_c_width_mm: 112.0
      panel_e_width_mm: 54.0
      col_gap_mm: 5.0
      row_gap_mm: 4.0
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
      composite_height_mm: 160.0
      composite_grid:
        left: 0.055
        right: 0.945
        bottom: 0.065
        top: 0.955
        hspace: 0.22
        wspace: 0.0
        width_ratios: [1.0]
        height_ratios: [32.0, 56.0, 62.0]
      split:
        standalone:
          b:
            width_mm: 171.0
            height_mm: 32.0
            subplots_adjust: {left: 0.06, right: 0.985, bottom: 0.16, top: 0.93}
          c:
            width_mm: 112.0
            height_mm: 56.0
            subplots_adjust: {left: 0.09, right: 0.98, bottom: 0.14, top: 0.92}
          d:
            width_mm: 171.0
            height_mm: 62.0
            subplots_adjust: {left: 0.06, right: 0.985, bottom: 0.14, top: 0.92}
          e:
            width_mm: 54.0
            height_mm: 56.0
            subplots_adjust: {left: 0.18, right: 0.97, bottom: 0.12, top: 0.92}
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
│   (a) Setup photo       │   (b) Physical principle   │  │ 65.0 mm
│   89.0 × 65.0 mm       │   89.0 × 65.0 mm          │  │ (external)
│                         │                           │  │
├─────────────────────────┴───────────────────────────┤ ─┼─  ← 5mm compose gap
│                                                     │  │
│  Generated strip: 183.0 × 65.0 mm                   │  │
│  GridSpec 1×3, wspace=0.40                           │  │
│  Margins: L=0.06 R=0.97 B=0.18 T=0.88               │  │
│  Usable: 166.5 × 45.5 mm                            │  │
│                                                     │  │ 65.0 mm
│  ┌──────────────┬──────────────┬──────────────┐      │  │ (generated)
│  │(c) Input→Out │(d) Repeat.   │(e) Directiv. │      │  │
│  │ 5 angles     │ mean±std     │ polar, 4band │      │  │
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

## Figure 4 — Solver Mechanism (5 panels)

Composed: hero architecture panel `a` in a decorated top slot plus generated
`b/c` portrait mechanism panels in the lower row, with the rightmost lower slot
split into a larger aggregation bridge `d` and a compact ablation closure `e`.
This contract adds an explicit manuscript collar and a dedicated panel-label
lane so Fig. 4 uses the same paper-facing visual discipline as Figs. 1, 3, and 5.

```
                         183.0 mm
┌──────────────────────────────────────────────────────┐
│                4.0 mm manuscript collar              │
│  ┌────────────────────────────────────────────────┐  │
│  │ (a) Architecture diagram                       │  │ 74.0 mm slot
│  │              175.0 × 74.0 mm                  │  │
│  └────────────────────────────────────────────────┘  │
│                    4.0 mm row gap                    │
│  ┌───────────────┬───────────────┬───────────────┐  │
│  │ (b) Routing   │ (c) Gated     │ (d) Aggrega-  │  │
│  │ formation     │ update +      │ tion bridge   │  │ 43.0 mm
│  │ 55.7 × 68.0   │ residual      │ 55.7 × 43.0   │  │
│  │               │ 55.7 × 68.0   ├───────────────┤  │ 3.0 mm gap
│  │               │               │ (e) Routing-  │  │ 22.0 mm
│  │               │               │ mechanism     │  │
│  │               │               │ ablation      │  │
│  │               │               │ 55.7 × 22.0   │  │
│  └───────────────┴───────────────┴───────────────┘  │
│                4.0 mm manuscript collar              │
└──────────────────────────────────────────────────────┘
                              total = 154.0 mm
```

| Parameter      | Value                                  |
|----------------|----------------------------------------|
| width_mm       | 183                                    |
| height_mm      | **154**                                |
| Outer collar   | **4.0 mm** on all four sides           |
| Panel (a) slot | **175.0 × 74.0 mm**                    |
| Row gap        | **4.0 mm**                             |
| Panels (b,c)   | **55.7 × 68.0 mm**                     |
| Panel (d)      | **55.7 × 43.0 mm**                     |
| Panel (e)      | **55.7 × 22.0 mm**                     |
| d/e gap        | **3.0 mm**                             |
| Column gap     | **4.0 mm** between lower panels        |
| Label lane     | **4.5 mm** at the top of each slot     |
| Content inset  | **1.5 mm** left/right/bottom           |

---

## Figure 5 — Performance + Structure (5 panels)

Full generated figure rebuilt as a compact two-row journal layout with
decorated-bbox budgeting, not raw-slot-only packing:

- Top row `a-b-c`: benchmark + unified confusion family + structure anchor
- Bottom row `d-e`: four-angle conditional outputs + per-angle benchmark

Panel `b` remains the dominant confusion-family block. `c` keeps its own
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
│  ┌──────────────┬────────────────────────┬───────┐   │  │
│  │ (a)          │ (b) unified confusion  │ (c)   │   │  │
│  │ benchmark    │ family block           │ H/QK  │   │  │ 62 mm
│  │ 48 × 62 mm   │ 73 × 62 mm             │ 40×62 │   │  │
│  └──────────────┴────────────────────────┴───────┘   │  │
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
| Top row               | 1×3, width ratios 48 / 73 / 40             |
| Bottom row            | 1×2, width ratios 90 / 72                  |
| Margins               | 7 mm on all sides                          |
| Row gap               | 16 mm                                      |
| Panel (a)             | 48 × 62 mm                                 |
| Panel (b)             | 73 × 62 mm                                 |
| Panel (c)             | 40 × 62 mm                                 |
| Panel (d)             | 90 × 64 mm                                 |
| Panel (e)             | 72 × 64 mm                                 |

---

## Figure 6 — Universality (5 panels)

Composed: a cropped manual exemplar strip `a` on top, a full-width generated `H` row `b`, then a shared row with low-rank continuity `c` on the left and screening consequence `e` on the right, followed by a full-width material-frequency mechanism row `d`.
Panel `a` remains a manual support asset and must stay tightly cropped to suppress the oversized slide-style title and bracket text from the legacy source image.
Panels `b-e` are data-backed and generated at their final manuscript slot sizes.

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────────────────────────────┐ ─┬─
│                outer top margin 4 mm                │  │
├──────────────────────────────────────────────────────┤ ─┼─
│      (a) 171.0 × 32.0 mm exemplar strip             │  │ 32.0 mm
├──────────────────────────────────────────────────────┤ ─┼─ 4.0 mm
│      (b) 171.0 × 32.0 mm cross-material H row       │  │ 32.0 mm
├───────────────────────────────┬──────────────────────┤ ─┼─ 4.0 mm
│ (c) 112.0 × 56.0 mm           │ (e) 54.0 × 56.0 mm  │  │ 56.0 mm
│ Low-rank continuity           │ Screening consequence│  │
├──────────────────────────────────────────────────────┤ ─┼─ 4.0 mm
│      (d) 171.0 × 62.0 mm material mechanism row     │  │ 62.0 mm
└───────────────────────────────┴──────────────────────┘ ─┼─
                  5.0 mm gap                           │  │
                                                        ─┴─ 4.0 mm bottom margin
```

| Parameter         | Value                       |
|------------------|-----------------------------|
| width_mm         | 183                         |
| height_mm        | **202**                     |
| outer_margin_x   | **6.0 mm** each side        |
| outer_margin_top | **4.0 mm**                  |
| outer_margin_bot | **4.0 mm**                  |
| Panel (a)        | **171.0 × 32.0 mm**         |
| Panel (b)        | **171.0 × 32.0 mm**         |
| Panel (c)        | **112.0 × 56.0 mm**         |
| Panel (d)        | **171.0 × 62.0 mm**         |
| Panel (e)        | **54.0 × 56.0 mm**          |
| row_gap          | **4.0 mm**                  |
| col_gap          | **5.0 mm**                  |
