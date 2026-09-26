"""Shared matplotlib style: light surface, thin marks, recessive grid, fixed hue order."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots in fixed order (validated reference palette, light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 2.0,
    "legend.frameon": False, "svg.fonttype": "none", "legend.fontsize": 8.5, "figure.dpi": 110, "savefig.dpi": 160,
})


def label_end(ax, x, y, text, color, dx=4, dy=0, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                color=INK2, fontsize=8.5, va="center", ha=ha)


def note(fig, text, width=120):
    import textwrap
    fig.text(0.01, -0.01, "\n".join(textwrap.wrap(text, width)), fontsize=7.5, color=MUTED,
             ha="left", va="top")
