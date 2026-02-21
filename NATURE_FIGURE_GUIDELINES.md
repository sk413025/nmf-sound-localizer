# Nature Figure Guidelines

This document summarizes the technical specifications for generating figures compliant with Nature journal standards.

## 1. Figure Dimensions
Figures are sized to fit within column widths.
- **Single Column**: 89 mm (3.50 inches) wide.
- **Double Column**: 183 mm (7.20 inches) wide.
- **1.5 Column**: 120–136 mm wide.
- **Maximum Height**: 247 mm (9.72 inches).

## 2. Typography
- **Font Family**: Sans-serif only. Prefer **Arial** or **Helvetica**.
- **Font Sizes**:
  - **Base/Body**: 7–8 pt.
  - **Axes Labels**: 7–8 pt.
  - **Tick Labels**: 5–7 pt.
  - **Legends**: 5–7 pt.
  - **Panel Labels**: 8 pt, **bold**, lowercase (e.g., **a**, **b**, **c**).
- **Font Embedding**: All fonts must be embedded in vector files (Type 42 / TrueType).

## 3. Line Weights & Styles
- **Axes/Ticks**: 0.5 – 1.0 pt (Recommended: 0.8 pt).
- **Data Lines**: 0.5 – 1.0 pt.
- **Grid Lines**: Avoid if possible. If necessary, use 0.5 pt or thinner, transparent/gray.
- **Tick Marks**: Distinct, usually pointing outward or inward consistently.

## 4. Colors & Aesthetics
- **Background**: White (transparent backgrounds are discouraged for final submission).
- **Color Palette**: Use colorblind-friendly palettes (e.g., Viridis, ColorBrewer).
- **Contrast**: Ensure high contrast for text and data.

## 5. File Formats
- **Vector (Preferred)**: PDF or EPS.
  - Editable text.
  - No bitmap rasterization of text/lines.
- **Raster**: TIFF or JPEG.
  - **Resolution**: Minimum 300 dpi (600 dpi for line art).
  - **Compression**: LZW (for TIFF) or High Quality (for JPEG).

## 6. Implementation in Python (Matplotlib)
Use the `NA_matplotlib_guild.py` helper to enforce these settings automatically.

```python
import NA_matplotlib_guild as na_style

# 1. Set RC Params
na_style.set_nature_rcparams(base_fontsize=8)

# 2. Create Figure with exact mm dimensions
fig = na_style.make_figure(width_mm=89, height_mm=60)

# 3. Add Panel Label
ax = fig.add_subplot(1,1,1)
na_style.add_panel_label(ax, "a")

# 4. Save
na_style.save_outputs(fig, "figure_name")
```
