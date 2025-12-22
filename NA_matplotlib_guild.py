import matplotlib as mpl
import matplotlib.pyplot as plt

def mm_to_in(mm):
    return mm / 25.4

def set_nature_rcparams(base_fontsize=7):
    """
    Set matplotlib rcParams for Nature-style figures.
    Nature guide: 7pt (max 8pt) for text.
    """
    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize - 1, # 6pt for labels
        "xtick.labelsize": base_fontsize - 1, # 6pt for ticks
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,

        # Lines
        "axes.linewidth": 0.5, # Thinner lines for small figures
        "lines.linewidth": 0.75,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        
        # Layout
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        
        # PDF
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def make_figure(width_mm=88, height_mm=60):
    fig = plt.figure(figsize=(mm_to_in(width_mm), mm_to_in(height_mm)))
    return fig

def save_outputs(fig, out_prefix, dpi_tiff=300):
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{out_prefix}.tiff", dpi=dpi_tiff, bbox_inches="tight", pad_inches=0.02)
