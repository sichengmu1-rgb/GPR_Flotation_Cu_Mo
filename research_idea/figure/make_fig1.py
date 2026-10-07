#!/usr/bin/env python3
"""
Figure 1 - the two scales and the closure between them.

Drawn at 6.3 in wide, which is the width it prints at, so the fonts below
appear at the size they are set.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figstyle as fs

B = fs.BLUE
fs.FS_BODY, fs.FS_SMALL = 7.4, 6.6


def box(ax, x, y, w, h, title, body, fc, ec, tc="white", bc="#D6E4F0"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.0,rounding_size=2.0",
                                linewidth=1.1, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=fs.FS_BODY, color=tc, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, body, ha="center", va="center",
            fontsize=fs.FS_SMALL, color=bc, linespacing=1.5)


fig, ax = plt.subplots(figsize=(6.3, 3.1))
ax.set_xlim(0, 100)
ax.set_ylim(0, 114)
fs.clean_axes(ax)

box(ax, 1, 52, 29, 34, "Molecular scale",
    "candidate collectors;\nDFT and MD descriptors", B["l4"], B["l5"])
box(ax, 35.5, 52, 29, 34, "Learned closure",
    "variogram-based\nGaussian process", B["l6"], B["l7"])
box(ax, 70, 52, 29, 34, "Process scale",
    "interfacial state;\ngrade and recovery", B["l2"], B["l4"], tc=fs.TEXT, bc=fs.MUTED)

box(ax, 35.5, 8, 29, 20, "Operating variables",
    "dosage, pH, aeration", B["l1"], B["l4"], tc=fs.TEXT, bc=fs.MUTED)

for x0, x1 in [(30, 35.5), (64.5, 70)]:
    ax.add_patch(FancyArrowPatch((x0 + 0.5, 69), (x1 - 0.5, 69), arrowstyle="-|>",
                                 mutation_scale=11, color=B["l6"], lw=1.4))
ax.add_patch(FancyArrowPatch((50, 28.5), (50, 51), arrowstyle="-|>",
                             mutation_scale=11, color=B["l6"], lw=1.4))

# experimental closure, routed above the boxes
ax.add_patch(FancyArrowPatch((84.5, 86.5), (15.5, 86.5), arrowstyle="-|>",
                             mutation_scale=11, color=B["l6"], lw=1.3,
                             linestyle="--", shrinkA=0, shrinkB=0,
                             connectionstyle="arc3,rad=0.30", zorder=1))
ax.text(50, 106, "staged experiments", fontsize=fs.FS_SMALL, color=B["l6"],
        ha="center", va="center", style="italic")

fig.subplots_adjust(0, 0, 1, 1)
fs.save(fig, "fig1_loops")
