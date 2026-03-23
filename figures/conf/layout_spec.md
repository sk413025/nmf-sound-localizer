# Figure Layout Specification

Single source of truth for intended panel-mm layout parameters for the active six main-paper figures.
Generator hardcoded values **must** match this document.
Use `figures/output/*.layout.json` as realized-geometry checks, not as the design authority.

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

Composed: manual architecture panel `a` on the left + generated diagnostic stack `b/c/d` on the right.
This replaced the old top-strip layout because the architecture artwork needed more width and less caption-like clutter.

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────┬──────────────────────┐ ─┬─
│                              │                      │  │
│ (a) Architecture diagram     │ (b) Convergence     │  │ 26.0 mm
│ 112.0 × 90.0 mm              │ 66.0 × 26.0 mm      │  │
│ manual, cropped to remove    ├──────────────────────┤ ─┼─ 3.0 mm
│ embedded footer/caption      │ (c) Ablation        │  │ 32.0 mm
│                              │ 66.0 × 32.0 mm      │  │
│                              ├──────────────────────┤ ─┼─ 3.0 mm
│                              │ (d) Per-angle acc.  │  │ 26.0 mm
│                              │ 66.0 × 26.0 mm      │  │
└──────────────────────────────┴──────────────────────┘ ─┴─
            112.0 mm              5.0 mm      66.0 mm
```

| Parameter      | Value                                  |
|----------------|----------------------------------------|
| width_mm       | 183                                    |
| height_mm      | **90**                                 |
| Left panel     | **112.0 × 90.0 mm**                    |
| Right width    | **66.0 mm**                            |
| Horizontal gap | **5.0 mm**                             |
| Panel (b)      | **66.0 × 26.0 mm**                     |
| Panel (c)      | **66.0 × 32.0 mm**                     |
| Panel (d)      | **66.0 × 26.0 mm**                     |
| Vertical gap   | **3.0 mm** between `b/c` and `c/d`     |

---

## Figure 5 — Performance + Structure (6 panels)

Full generated figure. Panels `b/c/d` use shared-colorbar two-row stacks; panel `e` remains a two-row routing stack.

```
                         183.0 mm
├──────────────────────────────────────────────────────┤
┌──────────────────────────────────────────────────────┐ ─┬─
│  GridSpec 2×3, hspace=0.22, wspace=0.28              │  │
│  Margins: L=0.06 R=0.98 B=0.06 T=0.97               │  │
│  Usable: 168.4 × 154.7 mm                           │  │
│  col_width = 168.4 / (3 + 2×0.28) = 47.3 mm         │  │
│  row_height = 154.7 / (2 + 0.22) = 69.7 mm          │  │
│                                                      │  │
│  Row 1 — 69.7 mm                                     │  │
│  ┌──────────────┬──────────────┬──────────────┐      │  │
│  │(a) SNR sweep │(b) H vs QK   │(c) Sel. prob │      │  │
│  │47.3 × 69.7   │47.3 × 69.7   │47.3 × 69.7  │      │  │
│  │              │ inner 2×2    │ inner 2×2   │      │  │
│  │ single plot  │ width_ratios │ width_ratios│      │  │
│  │ 47.3 × 69.7  │ [1.0, 0.05]  │ [1.0, 0.05] │      │  │
│  │              │ h=0.12 w=0.05│ h=0.12 w=0.05│     │  │
│  │              │ plot col ≈   │ plot col ≈  │      │  │ 69.7 mm
│  │              │ 43.9 mm      │ 43.9 mm     │      │  │
│  │              │ sub_h ≈ 32.9 │ sub_h ≈ 32.9│      │  │
│  │              │ cbar ≈ 2.2mm │ cbar ≈ 2.2mm│      │  │
│  └──────────────┴──────────────┴──────────────┘      │  │
│  ├── 47.3 ──┤13.2├── 47.3 ──┤13.2├── 47.3 ──┤       │  │ 170.0 mm
│                                                      │  │
│        inter-row gap = 0.22 × 69.7 = 15.3 mm        │  │
│                                                      │  │
│  Row 2 — 69.7 mm                                     │  │
│  ┌──────────────┬──────────────┬──────────────┐      │  │
│  │(d) Confus.   │(e) Routing   │(f) Per-angle │      │  │
│  │47.3 × 69.7   │47.3 × 69.7   │47.3 × 69.7  │      │  │
│  │ inner 2×2    │ inner 2×1    │              │      │  │
│  │ w=[1.0,0.05] │ hspace=0.22  │ single plot  │      │  │
│  │ h=0.12 w=0.05│ sub_h ≈ 31.4 │ 47.3 × 69.7  │      │  │ 69.7 mm
│  │ plot col ≈   │ gap ≈ 6.9mm  │              │      │  │
│  │ 43.9 mm      │              │              │      │  │
│  │ sub_h ≈ 32.9 │              │              │      │  │
│  └──────────────┴──────────────┴──────────────┘      │  │
└──────────────────────────────────────────────────────┘ ─┴─
```

| Parameter             | Value                                |
|-----------------------|--------------------------------------|
| width_mm              | 183                                  |
| height_mm             | **170**                              |
| GridSpec              | 2×3, hspace=0.22, wspace=0.28       |
| Margins               | L=0.06, R=0.98, B=0.06, T=0.97      |
| Usable                | 168.4 × 154.7 mm                     |
| col_width             | **47.3 mm**                          |
| row_height            | **69.7 mm**                          |
| Inter-row gap         | 0.22 × 69.7 = **15.3 mm**           |
| wspace gap            | 0.28 × 47.3 = **13.2 mm**           |
| Panel (a)             | **47.3 × 69.7 mm** (single plot)    |
| Panels (b/c/d) outer  | **47.3 × 69.7 mm** each              |
| Panels (b/c/d) inner  | 2×2 with shared colorbar column      |
| Inner plot column     | **≈ 43.9 mm**                        |
| Inner heatmap height  | **≈ 32.9 mm** each, gap ≈ 3.9 mm     |
| Shared colorbar       | **≈ 2.2 mm** wide                    |
| Panel (e) outer       | **47.3 × 69.7 mm**                   |
| Panel (e) sub-panels  | **≈ 47.3 × 31.4 mm** each, gap 6.9mm |
| Panel (f)             | **47.3 × 69.7 mm** (single plot)    |

---

## Figure 6 — Universality (5 panels)

Composed: panoramic manual strips `a/b` on top, then manual `c` + generated `d/e` diagnostic block below.
Top manual panels are intentionally cropped to suppress slide-style internal titles and keep the explanatory narrative in the caption.

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
│ RMSE                 │ 112.0 × 37.0 mm              │  │
│ 66.0 × 78.0 mm       ├───────────────────────────────┤ ─┼─ 4.0 mm
│ manual portrait      │ (e) Band-resolved routing    │  │ 37.0 mm
│ panel                │ 112.0 × 37.0 mm              │  │
└──────────────────────┴───────────────────────────────┘ ─┴─
     66.0 mm            5.0 mm              112.0 mm
```

| Parameter    | Value                                    |
|--------------|------------------------------------------|
| width_mm     | 183                                      |
| height_mm    | **144**                                  |
| Panel (a)    | **183.0 × 24.0 mm**                      |
| Panel (b)    | **183.0 × 34.0 mm**                      |
| Row gaps     | **4.0 mm** between `a/b` and `b/bottom`  |
| Panel (c)    | **66.0 × 78.0 mm**                       |
| Right width  | **112.0 mm**                             |
| Bottom gap   | **5.0 mm** between `c` and `d/e` block   |
| Panel (d)    | **112.0 × 37.0 mm**                      |
| Panel (e)    | **112.0 × 37.0 mm**                      |
| d/e gap      | **4.0 mm**                               |
