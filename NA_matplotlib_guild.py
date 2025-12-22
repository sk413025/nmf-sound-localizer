import matplotlib as mpl
import matplotlib.pyplot as plt

def mm_to_in(mm):
    return mm / 25.4

def set_nature_rcparams(base_fontsize=8):
    mpl.rcParams.update({
        # 字體（sans-serif；優先 Arial/Helvetica）
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

        # 字級：以 8 pt 為基準（縮放後仍可讀）
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,

        # 線寬：0.5–1 pt
        "axes.linewidth": 0.8,
        "lines.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        # 白底、乾淨
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        # PDF/PS 字體嵌入（避免審稿端字體被替換）
        "pdf.fonttype": 42,  # TrueType
        "ps.fonttype": 42,
    })

def make_figure(width_mm=88, height_mm=60):
    fig = plt.figure(figsize=(mm_to_in(width_mm), mm_to_in(height_mm)))
    return fig

def add_panel_label(ax, label, x=0.0, y=1.02):
    # Nature 常見：小寫粗體 a, b, c...
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="left")

def save_outputs(fig, out_prefix, dpi_tiff=300):
    # 1) 向量：PDF（線圖/統計圖優先）
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")

    # 2) 點陣：TIFF（影像/投稿系統可能需要）
    fig.savefig(f"{out_prefix}.tiff", dpi=dpi_tiff, bbox_inches="tight")
