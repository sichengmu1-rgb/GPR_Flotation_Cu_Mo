#!/usr/bin/env python3
"""Figure 3 - the staged program, and the block that never enters model development."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import figstyle as fs

B = fs.BLUE
fs.FS_SMALL, fs.FS_TINY = 7.0, 6.5

STAGES = [
    ("Screening", "single-factor tests"),
    ("Factorial", "descriptors × operating"),
    ("Response\nsurface", "refine the optimum"),
    ("Reserved\nblock", "never fitted"),
    ("Validation\nrounds", "repeated rounds"),
]

fig, ax = plt.subplots(figsize=(6.3, 1.95))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
fs.clean_axes(ax)

XS = [10, 30, 50, 70, 90]
Y = 58

ax.plot([XS[0], XS[-1]], [Y, Y], color="#CBD8E3", lw=1.2, zorder=1)

for i, (name, sub) in enumerate(STAGES):
    x = XS[i]
    reserved = i >= 3
    ax.scatter([x], [Y], s=150, c=B["l6"] if reserved else B["l4"],
               edgecolors="white", linewidths=1.0, zorder=3)
    ax.text(x, Y, str(i + 1), ha="center", va="center", fontsize=fs.FS_TINY,
            color="white", weight="bold", zorder=4)
    ax.text(x, 40, name, ha="center", va="center", fontsize=fs.FS_SMALL,
            color=fs.TEXT, weight="bold", linespacing=1.35)
    ax.text(x, 15, sub, ha="center", va="center", fontsize=fs.FS_TINY,
            color=fs.MUTED)

# brackets over the two blocks
def bracket(x0, x1, label, color, dashed):
    ax.plot([x0, x0, x1, x1], [74, 80, 80, 74], color=color, lw=1.0,
            ls=":" if dashed else "-")
    ax.text((x0 + x1) / 2, 88, label, ha="center", va="center",
            fontsize=fs.FS_TINY, color=color, weight="bold")


bracket(XS[0], XS[2], "enters model development", B["l5"], False)
bracket(XS[3], XS[4], "never enters model development", B["l6"], True)

fig.subplots_adjust(0, 0, 1, 1)
fs.save(fig, "fig3_program")
