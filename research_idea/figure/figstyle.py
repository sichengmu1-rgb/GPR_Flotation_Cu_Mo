#!/usr/bin/env python3
"""
Shared blue-palette style for every figure in the research proposal.

Import this from each `make_*.py` so that colors, fonts, and output
handling stay consistent across figures.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Blue palette, light -> dark. Shared with the Gantt chart.
# ----------------------------------------------------------------------
BLUE = {
    "l1": "#DCE9F2",  # palest   - backgrounds / bands
    "l2": "#A8C8E0",  # pale     - conditional / auxiliary
    "l3": "#7BAFD4",  # light    - molecular scale
    "l4": "#4E8FBF",  # medium   - interfacial / experimental
    "l5": "#2E6A9E",  # deep     - model core
    "l6": "#1C4E7C",  # navy     - process scale
    "l7": "#12365A",  # darkest  - outputs / decisions
}

GRID = "#D8E2EA"
AXIS = "#4A5A66"
TEXT = "#1A2530"
EDGE = "#2E6A9E"
ACCENT = "#0F2E4C"
MUTED = "#7A8894"

# Font sizes tuned so figures print at ~native scale inside a 12pt
# elsarticle page (\textwidth is about 5.9 in for this class).
FS_TITLE = 7.8
FS_BODY = 6.6
FS_SMALL = 5.9
FS_TINY = 5.3


def save(fig, stem):
    """Write a vector PDF for LaTeX plus a PNG preview."""
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", transparent=False)
    fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight")
    print(f"wrote {stem}.pdf and {stem}.png")


def clean_axes(ax):
    """Strip every spine and tick - used by the schematic diagrams."""
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
