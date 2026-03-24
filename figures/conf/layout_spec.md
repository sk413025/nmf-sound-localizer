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
      height_mm: 128.0
      panel_a_height_mm: 88.0
      row_gap_mm: 4.0
      bottom_gap_mm: 4.0
      bottom_weights: {b: 1.0, c: 1.0, d: 1.0}
    generator:
      composite_width_mm: 183.0
      composite_height_mm: 70.0
      composite_grid: {left: 0.06, right: 0.98, bottom: 0.18, top: 0.88, wspace: 0.28}
      split:
        panel_slot_width_mm: 58.333
        panel_slot_height_mm: 36.0
        standalone_subplots:
          b: {left: 0.18, right: 0.96, bottom: 0.24, top: 0.90}
          c: {left: 0.28, right: 0.97, bottom: 0.24, top: 0.90}
          d: {left: 0.18, right: 0.97, bottom: 0.24, top: 0.90}
  fig05:
    generator:
      composite_width_mm: 183.0
      composite_height_mm: 170.0
      outer_grid:
        left: 0.055
        right: 0.985
        bottom: 0.055
        top: 0.965
        hspace: 0.20
        height_ratios: [0.94, 1.06]
      top_row:
        wspace: 0.36
        width_ratios: [1.15, 0.925, 0.925]
      bottom_row:
        wspace: 0.42
        width_ratios: [1.0, 1.0, 1.0]
      heatmap_stack: {hspace: 0.35, wspace: 0.06, colorbar_ratio: 0.08}
      routing_stack: {hspace: 0.32}
      standalone:
        a:
          width_mm: 183.0
          height_mm: 70.0
          subplots_adjust: {left: 0.10, right: 0.95, bottom: 0.15, top: 0.92}
        b:
          width_mm: 183.0
          height_mm: 120.0
          grid: {left: 0.08, right: 0.95, bottom: 0.08, top: 0.94, hspace: 0.24}
        c:
          width_mm: 183.0
          height_mm: 120.0
          grid: {left: 0.08, right: 0.95, bottom: 0.08, top: 0.94, hspace: 0.24}
        d:
          width_mm: 183.0
          height_mm: 70.0
          grid: {left: 0.08, right: 0.95, bottom: 0.15, top: 0.86, wspace: 0.20}
        e:
          width_mm: 183.0
          height_mm: 100.0
          grid: {left: 0.08, right: 0.95, bottom: 0.10, top: 0.92, hspace: 0.36}
        f:
          width_mm: 183.0
          height_mm: 70.0
          subplots_adjust: {left: 0.08, right: 0.95, bottom: 0.15, top: 0.92}
  fig06:
    compose:
      width_mm: 183.0
      height_mm: 144.0
      panel_a_height_mm: 24.0
      panel_b_height_mm: 34.0
      bottom_height_mm: 78.0
      row_gap_mm: 4.0
      bottom_gap_mm: 5.0
      panel_c_width_mm: 62.0
      right_width_mm: 116.0
      de_height_mm: 37.0
      de_gap_mm: 4.0
    generator:
      composite_width_mm: 183.0
      composite_height_mm: 72.0
      composite_grid:
        left: 0.065
        right: 0.98
        bottom: 0.18
        top: 0.88
        wspace: 0.30
        width_ratios: [1.0, 1.28]
      split:
        panel_slot_width_mm: 116.0
        panel_slot_height_mm: 37.0
        standalone_subplots:
          d: {left: 0.11, right: 0.985, bottom: 0.24, top: 0.90}
          e: {left: 0.10, right: 0.985, bottom: 0.35, top: 0.90}
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

## Figure 4 — Solver Dynamics (4 panels)

Composed: full-width manual architecture panel `a` on top + generated `b/c/d` diagnostics across the bottom row.
The previous left-right layout left excessive blank space because the architecture artwork is too wide for a `112 × 90 mm` portrait-ish container.
Bottom-row panels must be generated at final slot size; do not downscale `183 mm`-wide standalone panels into manuscript slots.

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────────────────────────────┐ ─┬─
│ (a) Architecture diagram                             │  │ 88.0 mm
│ 183.0 × 88.0 mm                                      │  │
│ manual, cropped to remove embedded footer/caption    │  │
├──────────────────────────────────────────────────────┤ ─┼─ 4.0 mm
│ ┌──────────────┬──────────────┬──────────────┐       │  │
│ │(b) Converg.  │ (c) Ablation │ (d) Per-angle│       │  │
│ │58.3 × 36.0   │ 58.3 × 36.0  │ 58.3 × 36.0  │       │  │ 36.0 mm
│ └──────────────┴──────────────┴──────────────┘       │  │
│    4.0 mm gap         4.0 mm gap                     │  │
└──────────────────────────────────────────────────────┘ ─┴─
                                          total = 128.0 mm
```

| Parameter      | Value                                  |
|----------------|----------------------------------------|
| width_mm       | 183                                    |
| height_mm      | **128**                                |
| Panel (a)      | **183.0 × 88.0 mm**                    |
| Row gap        | **4.0 mm**                             |
| Bottom row     | **36.0 mm** tall                       |
| Bottom panels  | **strict equal width**                 |
| Panel (b)      | **58.3 × 36.0 mm**                     |
| Panel (c)      | **58.3 × 36.0 mm**                     |
| Panel (d)      | **58.3 × 36.0 mm**                     |
| Column gaps    | **4.0 mm** between `b/c` and `c/d`     |

---

## Figure 5 — Performance + Structure (6 panels)

Full generated figure. The layout is intentionally split into two row-level
story blocks rather than one continuous comparison stream:

- Top row `a-b-c`: benchmark + physical/classical context
- Bottom row `d-e-f`: matched ablation diagnostics isolating learned routing

Panels `b/c/d` use shared-colorbar two-row stacks; panel `e` remains a two-row routing stack.

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────────────────────────────┐ ─┬─
│  Nested 2-row layout, outer hspace=0.20              │  │
│  Margins: L=0.055 R=0.985 B=0.055 T=0.965           │  │
│  Usable: 170.2 × 154.7 mm                           │  │
│  Top row width ratios: [1.15, 0.925, 0.925]         │  │
│  Bottom row width ratios: [1.0, 1.0, 1.0]           │  │
│  Top row carries benchmark + reference context      │  │
│  Bottom row carries matched-ablation diagnostics    │  │
│                                                      │  │
│  Top row — context block                             │  │
│  ┌──────────────┬──────────────┬──────────────┐      │  │
│  │(a) Benchmark │(b) H vs QK   │(c) Classical│      │  │
│  │wider slot    │structure     │reference    │      │  │
│  │              │ inner 2×2    │ inner 2×2   │      │  │
│  │ single plot  │ width_ratios │ width_ratios│      │  │
│  │ 50.0 × 71.6  │ [1.0, 0.08]  │ [1.0, 0.08] │      │  │
│  │              │ h=0.35 w=0.06│ h=0.35 w=0.06│     │  │
│  │              │ plot col ≈   │ plot col ≈  │      │  │ 71.6 mm
│  │              │ 41–42 mm     │ 41–42 mm    │      │  │
│  │              │ stack gap ≈  │ stack gap ≈ │      │  │
│  │              │ 9–10 mm      │ 9–10 mm     │      │  │
│  │              │ cbar ≈ 3.6mm │ cbar ≈ 3.6mm│      │  │
│  └──────────────┴──────────────┴──────────────┘      │  │
│  ├─ wider ─┤gap├─ context ─┤gap├─ context ─┤       │  │ 170.0 mm
│                                                      │  │
│          enlarged row gap visually splits blocks    │  │
│                                                      │  │
│  Bottom row — matched ablation block                 │  │
│  ┌──────────────┬──────────────┬──────────────┐      │  │
│  │(d) Confus.   │(e) Routing   │(f) Per-angle │      │  │
│  │solver vs     │solver vs     │solver vs    │      │  │
│  │router-bypass │router-bypass │router-bypass│      │  │
│  │ inner 2×2    │ inner 2×1    │              │      │  │
│  │ w=[1.0,0.08] │ hspace=0.32  │ single plot  │      │  │
│  │ h=0.35 w=0.06│ sub_h ≈ 31–32│ 44–45 × 71.6 │      │  │ 71.6 mm
│  │ plot col ≈   │ gap ≈ 10 mm  │              │      │  │
│  │ 41–42 mm     │              │              │      │  │
│  └──────────────┴──────────────┴──────────────┘      │  │
└──────────────────────────────────────────────────────┘ ─┴─
```

| Parameter             | Value                                |
|-----------------------|--------------------------------------|
| width_mm              | 183                                  |
| height_mm             | **170**                              |
| Outer rows            | 2×1, hspace=0.20, height ratios 0.94 / 1.06 |
| Top row               | 1×3, width ratios 1.15 / 0.925 / 0.925 |
| Bottom row            | 1×3, equal-width matched-ablation panels |
| Margins               | L=0.055, R=0.985, B=0.055, T=0.965  |
| Usable                | 170.2 × 154.7 mm                     |
| Panel (a)             | widest slot in the figure            |
| Panels (b/c) outer    | smaller context/reference slots      |
| Panels (b/c/d) inner  | 2×2 with shared colorbar column and increased stack clearance |
| Inner plot column     | shared-colorbar heatmap stacks       |
| Shared colorbar       | retained for panels b/c/d            |
| Panel (e) outer       | equal-width routing stack with increased row spacing |
| Panel (f)             | equal-width summary line plot        |

---

## Figure 6 — Universality (5 panels)

Composed: panoramic manual strips `a/b` on top, then manual `c` + generated `d/e` diagnostic block below.
Top manual panels are intentionally cropped to suppress slide-style internal titles and keep the explanatory narrative in the caption.
Panels `d/e` must be generated at their final `116 × 37 mm` manuscript slot size; do not downscale `183 mm`-wide standalone diagnostics into the right-hand block.

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────────────────────────────┐ ─┬─
│ (a) Material exemplars strip                         │  │ 24.0 mm
│ 183.0 × 24.0 mm                                      │  │
├──────────────────────────────────────────────────────┤ ─┼─ 4.0 mm
│ (b) Representative heatmaps strip                    │  │ 34.0 mm
│ 183.0 × 34.0 mm                                      │  │
├──────────────────────┬───────────────────────────────┤ ─┼─ 4.0 mm
│ (c) Cross-material   │ (d) Per-band SVD             │  │ 37.0 mm
│ RMSE                 │ 116.0 × 37.0 mm              │  │
│ 62.0 × 78.0 mm       ├───────────────────────────────┤ ─┼─ 4.0 mm
│ manual portrait      │ (e) Band-resolved routing    │  │ 37.0 mm
│ panel                │ 116.0 × 37.0 mm              │  │
└──────────────────────┴───────────────────────────────┘ ─┴─
     62.0 mm            5.0 mm              116.0 mm
```

| Parameter    | Value                                    |
|--------------|------------------------------------------|
| width_mm     | 183                                      |
| height_mm    | **144**                                  |
| Panel (a)    | **183.0 × 24.0 mm**                      |
| Panel (b)    | **183.0 × 34.0 mm**                      |
| Row gaps     | **4.0 mm** between `a/b` and `b/bottom`  |
| Panel (c)    | **62.0 × 78.0 mm**                       |
| Right width  | **116.0 mm**                             |
| Bottom gap   | **5.0 mm** between `c` and `d/e` block   |
| Panel (d)    | **116.0 × 37.0 mm**                      |
| Panel (e)    | **116.0 × 37.0 mm**                      |
| d/e gap      | **4.0 mm**                               |
