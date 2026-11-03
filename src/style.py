"""
Figure style, copied from the thesis project (cracolandia/thesis/plots.py):
palette, left-aligned bold titles, no top/right spines, light y grid,
grey source note, and save_figure with the OneDrive retry.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Thesis palette: black and grey first, then the image colors. Contrast on white: black 17:1,
# grey 4.4:1, blue 3.4:1, taupe 3.1:1 (lines and markers); light blue and cream only as fills.
BLACK, DARK, GREY, LIGHT_GREY = "#1d1d1b", "#333333", "#7a7a75", "#b5b5b0"
BLUE, TAUPE, LIGHT_BLUE, CREAM = "#6C93A3", "#A3917C", "#CEE6F0", "#F0E0CE"
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", CREAM, TAUPE, "#5e5040", BLACK])
EMPTY, INK, MUTED, BG = "#e9e9e6", BLACK, "#55554f", "#ffffff"
NOTE = "#6b6b66"
GRID = "#ececea"


def style_axis(ax):
    """Light y grid, no top/right spines."""
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def titles(fig, title, subtitle=None, note=None):
    """Bold left-aligned title (subtitle in parentheses below) and grey source note."""
    text = title if subtitle is None else f"{title}\n({subtitle})"
    fig.suptitle(text, x=0.01, ha="left", fontweight="bold", fontsize=12)
    if note:
        fig.text(0.01, 0.008, note, fontsize=7.6, color=NOTE)


def save_figure(fig, path: Path, tries: int = 5) -> None:
    """Save PNG (dpi 170) and PDF, then close. Retries when Windows/OneDrive holds the old
    file open (OSError or PermissionError while it is open in a viewer or being synced)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for target in (path.with_suffix(".png"), path.with_suffix(".pdf")):
        for t in range(tries):
            try:
                fig.savefig(target, dpi=170, facecolor=BG)
                break
            except OSError as e:
                if t == tries - 1:
                    alt = target.with_name(target.stem + "_new" + target.suffix)
                    print(f"Could not overwrite {target.name} ({e}); close it in any viewer. Saved as {alt.name}")
                    fig.savefig(alt, dpi=170, facecolor=BG)
                else:
                    time.sleep(3)
    plt.close(fig)
    print("Saved", path.with_suffix(".png").name)
