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
      height_mm: 132.0
      panel_a_height_mm: 24.0
      panel_b_height_mm: 36.0
      bottom_height_mm: 64.0
      row_gap_mm: 4.0
      bottom_gap_mm: 5.0
      panel_c_width_mm: 112.0
      panel_d_width_mm: 66.0
    generator:
      composite_width_mm: 183.0
      composite_height_mm: 104.0
      composite_grid:
        left: 0.07
        right: 0.985
        bottom: 0.10
        top: 0.94
        hspace: 0.30
        wspace: 0.28
        width_ratios: [1.7, 1.0]
        height_ratios: [1.0, 1.55]
      split:
        standalone:
          b:
            width_mm: 183.0
            height_mm: 36.0
            subplots_adjust: {left: 0.06, right: 0.985, bottom: 0.22, top: 0.86}
          c:
            width_mm: 112.0
            height_mm: 64.0
            subplots_adjust: {left: 0.18, right: 0.98, bottom: 0.12, top: 0.95}
          d:
            width_mm: 66.0
            height_mm: 64.0
            subplots_adjust: {left: 0.20, right: 0.98, bottom: 0.12, top: 0.95}
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

## Figure 6 — Universality (4 panels)

Composed: a cropped manual exemplar strip `a` on top, a full-width generated H-matrix strip `b` beneath it, and a generated screening block `c/d` on the bottom row.
Panel `a` remains a manual support asset and must stay tightly cropped to suppress the oversized slide-style title and bracket text from the legacy source image.
Panels `b-d` are data-backed and must be generated at their final manuscript slot sizes; do not downscale a wide standalone plot into the smaller right-hand interpretation slot.

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────────────────────────────┐ ─┬─
│ (a) Material exemplars strip                         │  │ 24.0 mm
│ 183.0 × 24.0 mm                                      │  │
├──────────────────────────────────────────────────────┤ ─┼─ 4.0 mm
│ (b) Cross-material H matrices                        │  │ 36.0 mm
│ 183.0 × 36.0 mm                                      │  │
├───────────────────────────────┬──────────────────────┤ ─┼─ 4.0 mm
│ (c) Material screening        │ (d) Physics versus  │  │ 64.0 mm
│ metrics                       │ task accuracy        │  │
│ 112.0 × 64.0 mm               │ 66.0 × 64.0 mm       │  │
└───────────────────────────────┴──────────────────────┘ ─┴─
       112.0 mm                  5.0 mm     66.0 mm
```

| Parameter    | Value                                    |
|--------------|------------------------------------------|
| width_mm     | 183                                      |
| height_mm    | **132**                                  |
| Panel (a)    | **183.0 × 24.0 mm**                      |
| Panel (b)    | **183.0 × 36.0 mm**                      |
| Row gaps     | **4.0 mm** between `a/b` and `b/bottom`  |
| Panel (c)    | **112.0 × 64.0 mm**                      |
| Panel (d)    | **66.0 × 64.0 mm**                       |
| Bottom gap   | **5.0 mm** between `c` and `d`           |
