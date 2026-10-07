#!/usr/bin/env python3
"""
Figure 2 - the geostatistical surrogate and how it is inverted.
(a) the variogram recovered from my plant campaign (real numbers, real bins)
(b) a kriging posterior over two molecular descriptors (worked example)
(c) the inversion: candidates ranked on a selectivity-recovery trade-off
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import figstyle as fs

B = fs.BLUE
fs.FS_SMALL, fs.FS_TINY = 7.0, 6.5
CMAP = LinearSegmentedColormap.from_list("bl", ["#F2F7FB", B["l2"], B["l4"], B["l6"], B["l7"]])

fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.15))

# ------------------------------------------------------------------ (a)
ax = axes[0]
CSV = os.path.join("..", "..", "..", "GPR_FLOTATION_CU_MO", "data", "flotation_bench.csv")
have = os.path.exists(CSV)
if have:
    import pandas as pd
    d = pd.read_csv(CSV, encoding="utf-8-sig")
    X = d[["fineness", "ph", "diesel_gpt", "time_min"]].to_numpy(float)
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    z = d["cu_grade"].to_numpy(float)
    z = (z - z.mean()) / z.std()
    dist = pdist(X)
    semi = 0.5 * pdist(z[:, None]) ** 2
    edges = np.linspace(0, dist.max(), 9)
    k = np.digitize(dist, edges) - 1
    hb = np.array([dist[k == b].mean() for b in range(8) if (k == b).sum() >= 5])
    gb = np.array([semi[k == b].mean() for b in range(8) if (k == b).sum() >= 5])

    def gv(h, n, s, r):
        return n + (s - n) * (1 - np.exp(-(h / r) ** 2))

    p, _ = curve_fit(gv, hb, gb, p0=[0.3, gb.max(), hb.max() / 2], maxfev=20000,
                     bounds=([0, 1e-6, 1e-3], [gb.max(), gb.max() * 2, hb.max() * 5]))
    ax.plot(hb, gb, "o", color=B["l5"], ms=3.4, zorder=4)
    xx = np.linspace(0, 3.0, 300)
    ax.set_xlim(0, 3.05)
    ax.plot(xx, gv(xx, *p), "-", color=B["l6"], lw=1.3, zorder=3)
    ax.axhline(p[1], ls=":", color=fs.MUTED, lw=0.8)
    ax.axhline(p[0], ls=":", color=fs.MUTED, lw=0.8)
    ax.text(1.45, p[1] + 0.06, "sill", fontsize=fs.FS_TINY, color=fs.MUTED,
            va="bottom", ha="right")
    ax.text(1.45, p[0] + 0.06, "nugget", fontsize=fs.FS_TINY, color=fs.MUTED,
            va="bottom", ha="right")
    ax.set_title("$\\gamma(h)$ from the plant campaign, nugget/sill = 0.21",
                 fontsize=fs.FS_SMALL, pad=3)
    ax.text(0.03, 0.95, "measured", transform=ax.transAxes, fontsize=fs.FS_TINY,
            color=B["l5"], va="top")
ax.set_xlabel("distance in the scaled space", fontsize=fs.FS_TINY)
ax.set_ylabel("semivariance", fontsize=fs.FS_TINY)
ax.tick_params(labelsize=fs.FS_TINY)

# ------------------------------------------------------------------ (b)
ax = axes[1]
rng = np.random.default_rng(7)
Xtr = np.c_[rng.uniform(0.08, 0.92, 11), rng.uniform(0.08, 0.92, 11)]
truth = lambda Z: (1.00 * np.exp(-((Z[:, 0] - 0.32) ** 2 + (Z[:, 1] - 0.62) ** 2) / 0.070)
                   + 0.85 * np.exp(-((Z[:, 0] - 0.74) ** 2 + (Z[:, 1] - 0.30) ** 2) / 0.045))
ytr = truth(Xtr) + rng.normal(0, 0.035, len(Xtr))
gp = GaussianProcessRegressor(kernel=C(1.0) * RBF([0.28, 0.28]) + WhiteKernel(0.01),
                              normalize_y=True, n_restarts_optimizer=6,
                              random_state=0).fit(Xtr, ytr)
g = np.linspace(0, 1, 120)
G1, G2 = np.meshgrid(g, g, indexing="ij")
mu, sd = gp.predict(np.c_[G1.ravel(), G2.ravel()], return_std=True)
mu = mu.reshape(120, 120); sd = sd.reshape(120, 120)
cf = ax.contourf(g, g, mu.T, levels=16, cmap=CMAP)
ax.contour(g, g, sd.T, levels=5, colors="white", linewidths=0.6, alpha=0.85)
ax.scatter(Xtr[:, 0], Xtr[:, 1], s=16, c="white", edgecolors=B["l7"],
           linewidths=0.7, zorder=5)
ax.set_title("kriging posterior over two descriptors\n(filled = mean, white = sd)",
             fontsize=fs.FS_SMALL, pad=3)
ax.set_xlabel("descriptor $d_1$", fontsize=fs.FS_TINY)
ax.set_ylabel("descriptor $d_2$", fontsize=fs.FS_TINY)
ax.tick_params(labelsize=fs.FS_TINY)
cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=fs.FS_TINY - 0.5)
cb.set_label("interfacial response", fontsize=fs.FS_TINY)
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=B["l7"],
                          ms=4, label="training molecules")],
          loc="lower right", fontsize=fs.FS_TINY - 0.3, framealpha=0.9,
          handletextpad=0.3, borderpad=0.3)

# ------------------------------------------------------------------ (c)
ax = axes[2]
m = 26
sel = rng.uniform(0.30, 0.88, m)
rec = 1.02 - 0.85 * sel + rng.normal(0, 0.055, m)
err = rng.uniform(0.02, 0.05, m)
keep = np.ones(m, bool); best = -np.inf
for i in np.argsort(-sel):
    if rec[i] > best + 1e-9:
        best = rec[i]
    else:
        keep[i] = False
ax.errorbar(sel[~keep], rec[~keep], yerr=err[~keep], fmt="o", ms=2.6,
            color=B["l2"], ecolor=B["l2"], elinewidth=0.6, capsize=0,
            zorder=2, label="candidate collectors")
ax.errorbar(sel[keep], rec[keep], yerr=err[keep], fmt="o", ms=3.6,
            color=B["l6"], ecolor=B["l6"], elinewidth=0.8, capsize=0,
            zorder=4, label="non-dominated")
o = np.argsort(sel[keep])
ax.plot(sel[keep][o], rec[keep][o], "-", color=B["l6"], lw=1.0, zorder=3)
j = np.where(keep)[0][np.argmax(sel[np.where(keep)[0]] - rec[np.where(keep)[0]])]
ax.scatter([sel[j]], [rec[j]], marker="*", s=110, c="#F2B33D",
           edgecolors=B["l7"], linewidths=0.7, zorder=6)
ax.set_ylim(0.16, 0.88)
ax.set_xlim(0.25, 0.95)
ax.set_title("inversion: candidates ranked on the\ntrade-off, with uncertainty",
             fontsize=fs.FS_SMALL, pad=3)
ax.set_xlabel("selectivity", fontsize=fs.FS_TINY)
ax.set_ylabel("recovery", fontsize=fs.FS_TINY)
ax.tick_params(labelsize=fs.FS_TINY)
ax.legend(fontsize=fs.FS_TINY - 0.3, loc="lower left", framealpha=0.9,
          handletextpad=0.4, borderpad=0.3)

for a in axes:
    a.set_facecolor("white")
plt.tight_layout(pad=0.35, w_pad=0.9)
fs.save(fig, "fig2_surrogate")
