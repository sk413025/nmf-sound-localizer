# Figure Layout Specification

Single source of truth for all figure layout parameters.
Generator hardcoded values **must** match this document.

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

## Unified Font Specification

```yaml
fonts:
  panel_label:    8pt bold      # a, b, c, ...
  title:          6.5pt         # panel title
  axis_label:     6pt           # xlabel, ylabel
  tick_label:     5.5pt         # xtick, ytick
  legend:         6pt           # min 5.5pt
  annotation:     6pt           # stat text, d=1.95
  colorbar_tick:  5pt           # colorbar numbers
  colorbar_label: 5pt           # only if space permits
```

All font sizes **≥ 5pt** (Nature Comm minimum).

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

## Figure 1 — Paradigm Shift (5 panels)

Composed: external a,b (top) + generated c,d,e (bottom strip)

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌─────────────────────────┬───────────────────────────┐ ─┬─
│                         │                           │  │
│   (a) Setup photo       │   (b) Conceptual schem.   │  │ 65.0 mm
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
│  │(c) Waveforms │(d) Spectra   │(e) Control   │      │  │
│  │ 43.8 × 45.5  │ 43.8 × 45.5  │ 43.8 × 45.5 │      │  │
│  │              │              │              │      │  │
│  │  plot area:  │  plot area:  │  plot area:  │      │  │
│  │  38 × 40 mm  │  38 × 40 mm  │  38 × 40 mm │      │  │
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

## Figure 2 — SVD Spectrum (7 panels)

Full generated figure (no external assets).

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
                                                        ─┬─ top margin 8.5mm
┌──────────────────────────────────────────────────────┐  │
│  GridSpec 3×6, hspace=0.45, wspace=0.55              │  │
│  Margins: L=0.07 R=0.96 B=0.06 T=0.95               │  │
│  Usable: 162.9 × 151.3 mm                           │  │
│                                                      │  │
│  Row 1 (h_ratio=0.9) — height 40.2 mm               │  │
│  ┌──────────────────────────────────────────────┐    │  │
│  │ (a) SVD spectrum + cumulative energy         │    │  │ 40.2 mm
│  │ span: cols 1:5 → 105.1 × 40.2 mm            │    │  │
│  │ [4 cols: 4×18.6 + 3×(0.55×18.6) = 105.1mm]  │    │  │
│  └──────────────────────────────────────────────┘    │  │
│                    gap = 0.45 × 38.1 = 17.1 mm       │  │
│  Row 2 (h_ratio=0.85) — height 38.1 mm              │  │ 170.0 mm
│  ┌────────────┬────────────┬────────────┐            │  │
│  │(b) Mode 1  │(c) Mode 2  │(d) Mode 3  │            │  │ 38.1 mm
│  │ 2 cols ea. │ 2 cols ea. │ 2 cols ea. │            │  │
│  │freq + polar│freq + polar│freq + polar│            │  │
│  │47.5 × 38.1 │47.5 × 38.1 │47.5 × 38.1│            │  │
│  │  ┌───┬───┐ │  ┌───┬───┐ │  ┌───┬───┐ │            │  │
│  │  │18.6│18.6│ │  │18.6│18.6│ │  │18.6│18.6│            │  │
│  │  │freq│pol│ │  │freq│pol│ │  │freq│pol│ │            │  │
│  └──┴───┴───┴─┴──┴───┴───┴─┴──┴───┴───┴─┘            │  │
│                    gap = 17.1 mm                     │  │
│  Row 3 (h_ratio=0.85) — height 38.1 mm              │  │
│  ┌────────────┬────────────┬────────────┐            │  │
│  │(e) H heat  │(f) Reconst │(g) Corr    │            │  │ 38.1 mm
│  │ cols 0:2   │ cols 2:4   │ cols 4:6   │            │  │
│  │47.5 × 38.1 │47.5 × 38.1 │47.5 × 38.1│            │  │
│  └────────────┴────────────┴────────────┘            │  │
└──────────────────────────────────────────────────────┘ ─┴─
```

| Parameter       | Value                                                    |
|-----------------|----------------------------------------------------------|
| width_mm        | 183                                                      |
| height_mm       | **170** (at max)                                         |
| GridSpec        | 3×6, h_ratios=[0.9,0.85,0.85], hspace=0.45, wspace=0.55 |
| Margins         | L=0.07, R=0.96, B=0.06, T=0.95                          |
| Usable          | 162.9 × 151.3 mm                                         |
| col_width       | 162.9 / (6 + 5×0.55) = **18.6 mm**                      |
| Row 1 height    | **40.2 mm** (ratio 0.9)                                  |
| Row 2,3 height  | **38.1 mm** each (ratio 0.85)                            |
| Inter-row gap   | 0.45 × 38.1 = **17.1 mm**                               |
| Panel (a) width | 4×18.6 + 3×10.2 = **105.1 mm**                          |
| Panel (b,c,d)   | 2×18.6 + 1×10.2 = **47.5 × 38.1 mm** each               |
| Panel (e,f,g)   | 2×18.6 + 1×10.2 = **47.5 × 38.1 mm** each               |
| Sub-col (freq/polar) | **18.6 mm** wide each, 10.2mm gap between         |

---

## Figure 3 — Fingerprint Discriminability (5 panels)

Full generated figure.

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────────────────────────────┐ ─┬─
│  GridSpec 2×6, hspace=0.55, wspace=0.80              │  │
│  Margins: L=0.06 R=0.97 B=0.08 T=0.92               │  │
│  Usable: 166.5 × 121.8 mm                           │  │
│  col_width = 166.5 / (6 + 5×0.80) = 16.7 mm         │  │
│  row_height = 121.8 / (2 + 0.55) = 47.8 mm          │  │
│                                                      │  │
│  Row 1 — 47.8 mm                                     │  │
│  ┌──────────────┬──────────────┬──────────────┐      │  │
│  │(a) Heatmap   │(b) Violin    │(c) Line      │      │  │
│  │ cols 0:2     │ cols 2:4     │ cols 4:6     │      │  │ 47.8 mm
│  │ 46.7 × 47.8  │ 46.7 × 47.8  │ 46.7 × 47.8 │      │  │
│  │              │              │              │      │  │
│  │ plot: ~40×42 │ plot: ~40×42 │ plot: ~40×42 │      │  │
│  └──────────────┴──────────────┴──────────────┘      │  │
│  ├── 46.7 mm ──┤13.3├── 46.7 ──┤13.3├── 46.7 ──┤    │  │ 145.0 mm
│                                                      │  │
│        inter-row gap = 0.55 × 47.8 = 26.3 mm        │  │
│                                                      │  │
│  Row 2 — 47.8 mm                                     │  │
│  ┌─────────────────────┬─────────────────────┐       │  │
│  │(d) Pairwise matrix  │(e) OMP accuracy     │       │  │
│  │ cols 0:3             │ cols 3:6            │       │  │ 47.8 mm
│  │ 76.7 × 47.8 mm      │ 76.7 × 47.8 mm     │       │  │
│  │                     │                     │       │  │
│  │  plot: ~70×42 mm    │  plot: ~70×42 mm    │       │  │
│  └─────────────────────┴─────────────────────┘       │  │
│  ├───── 76.7 mm ──────┤13.3├───── 76.7 mm ──────┤    │  │
└──────────────────────────────────────────────────────┘ ─┴─
```

| Parameter       | Value                                |
|-----------------|--------------------------------------|
| width_mm        | 183                                  |
| height_mm       | 145                                  |
| GridSpec        | 2×6, hspace=0.55, wspace=0.80       |
| Margins         | L=0.06, R=0.97, B=0.08, T=0.92      |
| Usable          | 166.5 × 121.8 mm                     |
| col_width       | **16.7 mm**                          |
| row_height      | **47.8 mm**                          |
| Inter-row gap   | 0.55 × 47.8 = **26.3 mm**           |
| Panel (a) 2cols | 2×16.7 + 0.80×16.7 = **46.7 mm** w  |
| Panel (b) 2cols | **46.7 mm** w                        |
| Panel (c) 2cols | **46.7 mm** w                        |
| Panel (d) 3cols | 3×16.7 + 2×0.80×16.7 = **76.7 mm** w|
| Panel (e) 3cols | **76.7 mm** w                        |
| wspace gap      | 0.80 × 16.7 = **13.3 mm**           |

---

## Figure 4 — Solver Dynamics (4 panels)

Composed: external a (top) + generated b,c,d (bottom strip)

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌─────────────────────────────────────────────────────┐ ─┬─
│                                                     │  │
│   (a) Architecture diagram (external JPG)           │  │ ~65.0 mm
│   183.0 × 65.0 mm                                   │  │ (external)
│                                                     │  │
├─────────────────────────────────────────────────────┤ ─┼─  ← 5mm gap
│                                                     │  │
│  Generated strip: 183.0 × 70.0 mm                   │  │
│  GridSpec 1×3, w_ratios=[1.0, 1.2, 1.0], wspace=0.45│  │
│  Margins: L=0.06 R=0.97 B=0.18 T=0.88               │  │
│  Usable: 166.5 × 49.0 mm                            │  │
│                                                     │  │ 70.0 mm
│  unit = 166.5 / (1.0+1.2+1.0 + 2×0.45) = 40.0 mm   │  │ (generated)
│                                                     │  │
│  ┌───────────┬──────────────┬───────────┐            │  │
│  │(b) Converg│(c) Ablation  │(d) Bar    │            │  │
│  │w=1.0      │w=1.2         │w=1.0      │            │  │
│  │40.0×49.0  │48.0×49.0     │40.0×49.0  │            │  │
│  │           │              │           │            │  │
│  │ plot area:│ plot area:   │ plot area: │            │  │
│  │ 34×43 mm  │ 42×43 mm     │ 34×43 mm  │            │  │
│  └───────────┴──────────────┴───────────┘            │  │
│  ├── 40.0 ──┤18.0├── 48.0 ──┤18.0├── 40.0 ──┤       │  │
└─────────────────────────────────────────────────────┘ ─┴─
                                          total ≈ 140.0 mm
```

| Parameter    | Value                                            |
|--------------|--------------------------------------------------|
| width_mm     | 183                                              |
| height_mm    | 70                                               |
| GridSpec     | 1×3, w_ratios=[1.0,1.2,1.0], wspace=0.45        |
| Margins      | L=0.06, R=0.97, B=0.18, T=0.88                  |
| Usable       | 166.5 × 49.0 mm                                  |
| unit         | 166.5 / (3.2 + 0.90) = **40.0 mm**              |
| Panel (b)    | 1.0 × 40.0 = **40.0 × 49.0 mm**                 |
| Panel (c)    | 1.2 × 40.0 = **48.0 × 49.0 mm**                 |
| Panel (d)    | 1.0 × 40.0 = **40.0 × 49.0 mm**                 |
| wspace gap   | 0.45 × 40.0 = **18.0 mm**                       |

---

## Figure 5 — Performance + Structure (6 panels)

Full generated figure. Panels b,c,d,e contain inner sub-panels.

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────────────────────────────┐ ─┬─
│  GridSpec 2×3, hspace=0.30, wspace=0.42              │  │
│  Margins: L=0.07 R=0.96 B=0.05 T=0.96               │  │
│  Usable: 162.9 × 154.7 mm                           │  │
│  col_width = 162.9 / (3 + 2×0.42) = 42.4 mm         │  │
│  row_height = 154.7 / (2 + 0.30) = 67.3 mm          │  │
│                                                      │  │
│  Row 1 — 67.3 mm                                     │  │
│  ┌──────────────┬──────────────┬──────────────┐      │  │
│  │(a) SNR sweep │(b) H vs QK   │(c) Sel. prob │      │  │
│  │42.4 × 67.3   │42.4 × 67.3   │42.4 × 67.3  │      │  │
│  │              │ inner 2×1    │ inner 2×1    │      │  │
│  │              │ hspace=0.20  │ hspace=0.20  │      │  │
│  │              │┌────────────┐│┌────────────┐│      │  │
│  │  single plot ││b1: H_corr  │││c1: OMP sel ││      │  │
│  │  42.4 × 67.3 ││42.4 × 30.6 │││42.4 × 30.6││      │  │
│  │              │├────────────┤│├────────────┤│      │  │ 67.3 mm
│  │              ││gap 6.1 mm  │││gap 6.1 mm  ││      │  │
│  │              │├────────────┤│├────────────┤│      │  │
│  │              ││b2: QK_corr │││c2: AI sel  ││      │  │
│  │              ││42.4 × 30.6 │││42.4 × 30.6││      │  │
│  │              │└────────────┘│└────────────┘│      │  │
│  └──────────────┴──────────────┴──────────────┘      │  │
│  ├── 42.4 ──┤17.8├── 42.4 ──┤17.8├── 42.4 ──┤       │  │ 170.0 mm
│                                                      │  │
│        inter-row gap = 0.30 × 67.3 = 20.2 mm        │  │
│                                                      │  │
│  Row 2 — 67.3 mm                                     │  │
│  ┌──────────────┬──────────────┬──────────────┐      │  │
│  │(d) Confus.   │(e) Routing   │(f) Per-angle │      │  │
│  │42.4 × 67.3   │42.4 × 67.3   │42.4 × 67.3  │      │  │
│  │ inner 2×1   │ inner 2×1    │              │      │  │
│  │ hspace=0.20  │ hspace=0.35  │              │      │  │
│  │┌────────────┐│┌────────────┐│              │      │  │
│  ││d1: BL CM   │││e1: @55°    ││  single plot │      │  │
│  ││42.4 × 30.6 │││42.4 × 28.6 ││  42.4 × 67.3│      │  │
│  │├────────────┤│├────────────┤│              │      │  │ 67.3 mm
│  ││gap 6.1 mm  │││gap 10.0 mm ││              │      │  │
│  │├────────────┤│├────────────┤│              │      │  │
│  ││d2: NT CM   │││e2: @100°   ││              │      │  │
│  ││42.4 × 30.6 │││42.4 × 28.6 ││              │      │  │
│  │└────────────┘│└────────────┘│              │      │  │
│  │              │              │              │      │  │
│  └──────────────┴──────────────┴──────────────┘      │  │
└──────────────────────────────────────────────────────┘ ─┴─

Sub-panel detail:

  Panel (b,c) — inner GridSpec 2×1, hspace=0.20
  ┌──────────────┐
  │ sub1          │ 42.4 × 30.6 mm
  │ (heatmap)     │   sub_h = 67.3 / (2+0.20) = 30.6 mm
  ├──────────────┤ ← gap = 0.20 × 30.6 = 6.1 mm
  │ sub2          │ 42.4 × 30.6 mm
  │ (heatmap)     │
  └──────────────┘

  Panel (d) — inner GridSpec 2×1, hspace=0.20
  ┌──────────────┐
  │ d1: BL CM     │ 42.4 × 30.6 mm
  │ (heatmap)     │   sub_h = 67.3 / (2+0.20) = 30.6 mm
  ├──────────────┤ ← gap = 0.20 × 30.6 = 6.1 mm
  │ d2: NT CM     │ 42.4 × 30.6 mm
  │ (heatmap)     │
  └──────────────┘

  Panel (e) — inner GridSpec 2×1, hspace=0.35
  ┌──────────────┐
  │ e1: @55°      │ 42.4 × 28.6 mm
  │ (bar chart)   │   sub_h = 67.3 / (2+0.35) = 28.6 mm
  ├──────────────┤ ← gap = 0.35 × 28.6 = 10.0 mm
  │ e2: @100°     │ 42.4 × 28.6 mm
  │ (bar chart)   │
  └──────────────┘
```

| Parameter             | Value                                |
|-----------------------|--------------------------------------|
| width_mm              | 183                                  |
| height_mm             | **170** (fixed from 185)             |
| GridSpec              | 2×3, hspace=0.30, wspace=0.42       |
| Margins               | L=0.07, R=0.96, B=0.05, T=0.96      |
| Usable                | 162.9 × 154.7 mm                     |
| col_width             | **42.4 mm**                          |
| row_height            | **67.3 mm**                          |
| Inter-row gap         | 0.30 × 67.3 = **20.2 mm**           |
| wspace gap            | 0.42 × 42.4 = **17.8 mm**           |
| Panel (a)             | **42.4 × 67.3 mm** (single plot)    |
| Panel (b) outer       | **42.4 × 67.3 mm**                   |
| Panel (b) sub-panels  | **42.4 × 30.6 mm** each, gap 6.1mm  |
| Panel (c) outer       | **42.4 × 67.3 mm**                   |
| Panel (c) sub-panels  | **42.4 × 30.6 mm** each, gap 6.1mm  |
| Panel (d) outer       | **42.4 × 67.3 mm**                   |
| Panel (d) sub-panels  | **42.4 × 30.6 mm** each, gap 6.1mm  |
| Panel (e) outer       | **42.4 × 67.3 mm**                   |
| Panel (e) sub-panels  | **42.4 × 28.6 mm** each, gap 10.0mm |
| Panel (f)             | **42.4 × 67.3 mm** (single plot)    |

---

## Figure 6 — Universality (5 panels)

Composed: external a,b,c (top) + generated d,e (bottom strip)

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌─────────────┬────────────────┬───────────────┐        ─┬─
│             │                │               │         │
│(a) Photos   │(b) Heatmaps   │(c) Box/RMSE   │         │ ~70.0 mm
│ 60×70 mm    │ 60×70 mm       │ 60×70 mm      │         │ (external)
│             │                │               │         │
├─────────────┴────────────────┴───────────────┤        ─┼─  ← 5mm gap
│                                              │         │
│  Generated strip: 183.0 × 70.0 mm            │         │
│  GridSpec 1×2, w_ratios=[1.0, 1.2], wspace=0.35│       │
│  Margins: L=0.07 R=0.97 B=0.18 T=0.88        │         │
│  Usable: 164.7 × 49.0 mm                     │         │
│                                              │         │ 70.0 mm
│  unit = 164.7 / (1.0+1.2 + 0.35) = 64.6 mm  │         │ (generated)
│                                              │         │
│  ┌───────────────┬──────────────────┐         │         │
│  │(d) SVD decay  │(e) Band routing  │         │         │
│  │w_ratio=1.0    │w_ratio=1.2       │         │         │
│  │64.6 × 49.0 mm │77.5 × 49.0 mm   │         │         │
│  │               │                  │         │         │
│  │ plot: ~58×43  │ plot: ~71×43 mm  │         │         │
│  └───────────────┴──────────────────┘         │         │
│  ├── 64.6 mm ──┤22.6├──── 77.5 mm ────┤       │         │
└──────────────────────────────────────────────┘        ─┴─
                                          total ≈ 145.0 mm
```

| Parameter    | Value                                    |
|--------------|------------------------------------------|
| width_mm     | 183                                      |
| height_mm    | 70                                       |
| GridSpec     | 1×2, w_ratios=[1.0,1.2], wspace=0.35    |
| Margins      | L=0.07, R=0.97, B=0.18, T=0.88          |
| Usable       | 164.7 × 49.0 mm                          |
| unit         | 164.7 / (2.2 + 0.35) = **64.6 mm**      |
| Panel (d)    | 1.0 × 64.6 = **64.6 × 49.0 mm**         |
| Panel (e)    | 1.2 × 64.6 = **77.5 × 49.0 mm**         |
| wspace gap   | 0.35 × 64.6 = **22.6 mm**               |
