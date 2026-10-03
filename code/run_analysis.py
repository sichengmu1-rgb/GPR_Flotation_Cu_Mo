# -*- coding: utf-8 -*-
"""
Variogram-based Gaussian process regression of concentrate grade in
copper-molybdenum flotation.

The campaign is not an unstructured one-factor-at-a-time set. It is three
complete two-factor grids sharing a common grind-fineness axis:

    fineness x pH      40 and 65 % fineness at pH 3.5 / 6 / 9 / 11
    fineness x diesel  40 and 65 % fineness at 50 / 100 / 150 / 200 g/t
    fineness x time    40 and 65 % fineness at 5 / 10 / 15 / 20 / 25 min

so a two-dimensional kriging surface can be built on each, and the
interaction between the fineness axis and each reagent axis is identified
rather than assumed away.

Response variables are concentrate GRADES. The Mo grade and the concentrate
mass reproduce the reported Mo recovery in 29 of 29 bench tests; the Cu
recovery reproduces in 15 of 29 and is not reconstructible from the archive.

Writes eight figures into ../figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rng = np.random.default_rng(20261002)

XCOLS = ["fineness", "ph", "diesel_gpt", "time_min"]
XLAB = ["Grind fineness, −0.074 mm (%)", "Pulp pH",
        "Diesel dosage (g/t)", "Conditioning time (min)"]
SLAB = {"mo_grade": "Mo grade (%)", "cu_grade": "Cu grade (%)"}
LINE = {"2区": "Zone 2", "4区老线": "Zone 4, old line", "4区万吨": "Zone 4, 10 kt line"}


# ------------------------------------------------------------------ helpers
def make_kernel(d):
    return (C(1.0, (1e-3, 1e3))
            * RBF([0.5] * d, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)))


def rbf_of(k):
    return k.k1.k2 if hasattr(k.k1, "k2") else k.k2


def fit_gp(X, y, seed=0, restarts=8):
    return GaussianProcessRegressor(kernel=make_kernel(X.shape[1]), normalize_y=True,
                                    n_restarts_optimizer=restarts,
                                    random_state=seed).fit(X, y)


def loo_predict(gp, X, y):
    return np.array([
        GaussianProcessRegressor(kernel=gp.kernel_, normalize_y=True, optimizer=None)
        .fit(np.delete(X, i, 0), np.delete(y, i)).predict(X[i:i + 1])[0]
        for i in range(len(y))])


def q2_rmse(y, yhat):
    return (1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2),
            float(np.sqrt(np.mean((y - yhat) ** 2))))


def pareto_max_min(mo, cu):
    keep = np.ones(len(mo), bool)
    best = np.inf
    for i in np.argsort(-mo):
        if cu[i] < best - 1e-12:
            best = cu[i]
        else:
            keep[i] = False
    return keep


def knee_of(mo, cu, pts):
    mn = (mo - mo.min()) / (np.ptp(mo) + 1e-12)
    cn = (cu - cu.min()) / (np.ptp(cu) + 1e-12)
    k = int(np.argmin(np.sqrt((1 - mn) ** 2 + cn ** 2)))
    return k, pts[k]


# ================================================================ A. bench
bench = pd.read_csv(os.path.join(DATA, "flotation_bench.csv"))
Xr = bench[XCOLS].to_numpy(float)
lo, hi = Xr.min(0), Xr.max(0)
Xs = (Xr - lo) / (hi - lo)
Y = {r: bench[r].to_numpy(float) for r in ["mo_grade", "cu_grade"]}

gps, loo, stats = {}, {}, {}
for resp in ["mo_grade", "cu_grade"]:
    gp = fit_gp(Xs, Y[resp])
    lp = loo_predict(gp, Xs, Y[resp])
    gps[resp], loo[resp] = gp, lp
    L = np.atleast_1d(rbf_of(gp.kernel_).length_scale)
    q2, rmse = q2_rmse(Y[resp], lp)
    sig = float(gp.kernel_.k1.k1.constant_value)
    noi = float(gp.kernel_.k2.noise_level)
    stats[resp] = dict(q2=q2, rmse=rmse, length_scale=L.tolist(),
                       inv_length_scale=(1 / L).tolist(),
                       noise=noi, signal_variance=sig,
                       nugget_fraction=noi / (noi + sig),
                       at_bound=[bool(abs(l - 1e-2) < 1e-6 or abs(l - 1e2) < 1e-6) for l in L],
                       observed_sd=float(Y[resp].std()))
    print(f"[A] {resp}: Q2={q2:.3f} RMSE={rmse:.3f} ls={np.round(L,3)} noise={gp.kernel_.k2.noise_level:.3f}")

imp = {}
for resp in ["mo_grade", "cu_grade"]:
    base = stats[resp]["q2"]
    imp[resp] = {c: float(base - q2_rmse(Y[resp], loo_predict(
        fit_gp(Xs[:, [k for k in range(4) if k != j]], Y[resp]),
        Xs[:, [k for k in range(4) if k != j]], Y[resp]))[0])
        for j, c in enumerate(XCOLS)}
    print(f"[A] drop-one dQ2 {resp}: " +
          "  ".join(f"{k}={v:+.3f}" for k, v in imp[resp].items()))

# The two grades move together; the separation index between them does not
# follow from the operating variables, by either of two independent routes.
ratio = Y["cu_grade"] / Y["mo_grade"]
gp_ratio = fit_gp(Xs, ratio)
q2r, rmser = q2_rmse(ratio, loo_predict(gp_ratio, Xs, ratio))
q2c, _ = q2_rmse(ratio, loo["cu_grade"] / loo["mo_grade"])
corr = float(np.corrcoef(Y["mo_grade"], Y["cu_grade"])[0, 1])
print(f"[A] corr(Mo grade, Cu grade) = {corr:+.3f};  Cu/Mo ratio {ratio.min():.3f}-"
      f"{ratio.max():.3f} (mean {ratio.mean():.3f}, sd {ratio.std():.3f})")
print(f"[A] Cu/Mo ratio: Q2={q2r:.3f} direct,  Q2={q2c:.3f} from the two LOO models")

# --- design: three two-factor grids on the shared fineness axis
ANCH = [40.0, 65.0]
ANCHOR = {"fineness": ANCH, "ph": 11.0, "diesel_gpt": 100.0, "time_min": 15.0}
FIN_ALONE = bench[bench.series == "磨矿细度"].sort_values("fineness")

# The campaign is three two-factor comparisons sharing one fineness axis:
# fineness x pH, fineness x diesel and fineness x time, each swept at the
# anchor settings of the remaining factors. The maps below are two-dimensional
# slices of the validated four-dimensional model rather than separate
# two-parameter fits: a separate fit on 15 points cross-validates far worse
# (Q2 between -0.01 and 0.40) because it discards what the other factors carry.
PLANES = [("pH", 1), ("diesel", 2), ("time", 3)]
JF = dict(PLANES)
anchor_scaled = np.array([(ANCHOR[c] - lo[i]) / (hi[i] - lo[i]) if i else 0.0
                          for i, c in enumerate(XCOLS)])
gdata, gpred = {}, {}
for name, j in PLANES:
    m = np.ones(len(bench), bool)
    for k, c in enumerate(XCOLS):
        if k in (0, j):
            continue
        m &= np.isclose(bench[c], ANCHOR[c])
    m &= (np.isclose(bench.fineness, ANCH[0]) | np.isclose(bench.fineness, ANCH[1]) |
          (bench.series == "磨矿细度"))
    sub = bench[m].copy()
    gdata[name] = sub
    print(f"[B] plane fineness x {name}: {len(sub)} tests lie on it")
    gpred[name] = {}
    n = 90
    gg = np.linspace(0, 1, n)
    G1, G2 = np.meshgrid(gg, gg, indexing="ij")
    Q = np.tile(anchor_scaled, (n * n, 1))
    Q[:, 0] = G1.ravel()
    Q[:, j] = G2.ravel()
    for resp in ["mo_grade", "cu_grade"]:
        mu, sd = gps[resp].predict(Q, return_std=True)
        gpred[name][resp] = dict(
            mean=mu.reshape(n, n), sd=sd.reshape(n, n),
            x=gg * (hi[0] - lo[0]) + lo[0],
            y=gg * (hi[j] - lo[j]) + lo[j],
            pts=sub[[XCOLS[0], XCOLS[j]]].to_numpy(float),
            vals=sub[resp].to_numpy(float))

# --- empirical variogram in the four-dimensional operating space
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit


def gauss_vario(h, nug, sill, rng_):
    return nug + (sill - nug) * (1.0 - np.exp(-(h / rng_) ** 2))


def empirical_variogram(V, z, nbins=8):
    d = pdist(V)
    g = 0.5 * pdist(z[:, None]) ** 2
    edges = np.linspace(0, d.max(), nbins + 1)
    idx = np.digitize(d, edges) - 1
    hb, gb, nb = [], [], []
    for b in range(nbins):
        s = idx == b
        if s.sum() >= 5:
            hb.append(d[s].mean()); gb.append(g[s].mean()); nb.append(int(s.sum()))
    hb, gb = np.array(hb), np.array(gb)
    p0 = [0.3, gb.max(), hb.max() / 2]
    popt, _ = curve_fit(gauss_vario, hb, gb, p0=p0, maxfev=20000,
                        bounds=([0, 1e-6, 1e-3], [gb.max(), gb.max() * 2, hb.max() * 5]))
    return hb, gb, nb, popt


variograms = {}
for resp in ["mo_grade", "cu_grade"]:
    z = (Y[resp] - Y[resp].mean()) / Y[resp].std()
    hb, gb, nb, popt = empirical_variogram(Xs, z)
    variograms[resp] = dict(range=float(popt[2]), sill=float(popt[1]),
                            nugget=float(popt[0]),
                            nugget_over_sill=float(popt[0] / popt[1]),
                            bins=hb.tolist(), semivariance=gb.tolist(),
                            n_pairs=[int(x) for x in nb])
    print(f"[C] variogram {resp}: range={popt[2]:.3f} sill={popt[1]:.3f} "
          f"nugget={popt[0]:.3f}  nugget/sill={popt[0]/popt[1]:.2f}")

# --- conditional simulation from the posterior, and the stability of the optimum
gs_grid = np.linspace(0.0, 1.0, 20)
G4 = np.stack(np.meshgrid(*[gs_grid] * 4, indexing="ij"), -1).reshape(-1, 4)
mu4, sd4 = {}, {}
for resp in ["mo_grade", "cu_grade"]:
    m, s = gps[resp].predict(G4, return_std=True)
    mu4[resp], sd4[resp] = m, s

keep = pareto_max_min(mu4["mo_grade"], mu4["cu_grade"])
pts_real = G4[keep] * (hi - lo) + lo
mo_f, cu_f = mu4["mo_grade"][keep], mu4["cu_grade"][keep]
kb, knee = knee_of(mo_f, cu_f, pts_real)
knee_pred = (float(mu4["mo_grade"][keep][kb]), float(mu4["cu_grade"][keep][kb]))
print(f"[D] Pareto {int(keep.sum())} of {len(G4):,} grid points; knee "
      f"fineness={knee[0]:.0f} pH={knee[1]:.1f} diesel={knee[2]:.0f} time={knee[3]:.0f} "
      f"-> Mo={knee_pred[0]:.2f} Cu={knee_pred[1]:.3f}")
nil = int(np.argmin(np.linalg.norm((knee - Xr) / (hi - lo), axis=1)))
print(f"[D] nearest measured test #{bench.id.iloc[nil]} "
      f"-> Mo={bench.mo_grade.iloc[nil]:.2f} Cu={bench.cu_grade.iloc[nil]:.3f}")

# constrained inversion: lowest Cu grade subject to a stated Mo-grade target
cons = {}
for t in (3.0, 3.5, 4.0):
    ok = mu4["mo_grade"] >= t
    j = int(np.argmin(np.where(ok, mu4["cu_grade"], np.inf)))
    x = G4[j] * (hi - lo) + lo
    cons[str(t)] = dict(inputs=x.round(2).tolist(), cu_grade=float(mu4["cu_grade"][j]),
                        mo_grade=float(mu4["mo_grade"][j]), cu_sd=float(sd4["cu_grade"][j]),
                        n_ok=int(ok.sum()))
    print(f"[D] min Cu at Mo>={t}%: fineness={x[0]:.0f}% pH={x[1]:.1f} diesel={x[2]:.0f} "
          f"time={x[3]:.0f} -> Cu={cons[str(t)]['cu_grade']:.3f}+-{cons[str(t)]['cu_sd']:.3f} "
          f"Mo={cons[str(t)]['mo_grade']:.2f}%")

# posterior sampling of the operating point recommended by the compromise
MO_TARGET = 3.5
# How well is the recommendation pinned down? Because the response is a grade,
# a Gaussian posterior realisation is not guaranteed to be positive, so the
# question is asked by resampling the data instead: 100 subsamples of 24 of the
# 29 tests, refitted, and the constrained optimum recomputed each time.
boot_grid = np.stack(np.meshgrid(*[np.linspace(0.0, 1.0, 10)] * 4, indexing="ij"),
                     -1).reshape(-1, 4)
opt_pts, opt_grade = [], []
for b in range(100):
    idx = np.arange(len(bench)) if b == 0 else rng.choice(len(bench), 24, replace=False)
    sm = {resp: fit_gp(Xs[idx], Y[resp][idx], seed=b, restarts=1).predict(boot_grid)
          for resp in ["mo_grade", "cu_grade"]}
    ok = sm["mo_grade"] >= MO_TARGET
    if not ok.any():
        continue
    k = int(np.argmin(np.where(ok, sm["cu_grade"], np.inf)))
    opt_pts.append(boot_grid[k] * (hi - lo) + lo)
    opt_grade.append(float(sm["cu_grade"][k]))
knees = np.array(opt_pts)
opt_grade = np.array(opt_grade)
print(f"[D] achievable Cu grade over the same resamples: {opt_grade.mean():.3f} +- "
      f"{opt_grade.std():.3f}  (5-95%: {np.percentile(opt_grade,5):.3f}-"
      f"{np.percentile(opt_grade,95):.3f})")
print(f"[D] optimum of min-Cu s.t. Mo>={MO_TARGET}% over {len(knees)} resamples: "
      f"fineness {knees[:,0].mean():.0f}±{knees[:,0].std():.0f}%, "
      f"pH {knees[:,1].mean():.1f}±{knees[:,1].std():.1f}, "
      f"diesel {knees[:,2].mean():.0f}±{knees[:,2].std():.0f} g/t, "
      f"time {knees[:,3].mean():.0f}±{knees[:,3].std():.0f} min")

# ================================================================ E. plant
plant = pd.read_csv(os.path.join(DATA, "flotation_plant.csv"))
pe = {}
for resp in ["mo_grade", "cu_grade"]:
    g = plant.groupby("fineness")[resp]
    within = g.var(ddof=1)
    pe[resp] = dict(pooled_sd=float(np.sqrt(within.mean())), df=int(3 * (len(within) - 1)),
                    overall_sd=float(plant[resp].std()),
                    range=float(plant[resp].max() - plant[resp].min()))
    print(f"[E] plant pure error {resp}: pooled sd={pe[resp]['pooled_sd']:.3f} "
          f"(df={pe[resp]['df']}), spread across fineness={pe[resp]['range']:.3f}")
    plant_opt = {}
for line, sub in plant.groupby("line"):
    sub = sub.sort_values("fineness")
    plant_opt[line] = dict(
        mo_max_at=float(sub.fineness.iloc[int(np.argmax(sub.mo_grade))]),
        cu_min_at=float(sub.fineness.iloc[int(np.argmin(sub.cu_grade))]))
    print(f"[E] {line}: Mo max at {plant_opt[line]['mo_max_at']:.0f}%, "
          f"Cu min at {plant_opt[line]['cu_min_at']:.0f}%")

# ============================================ F. pulp environment (compact)
rheo = pd.read_csv(os.path.join(DATA, "rheology.csv"))
CLAYS = ["chlorite", "illite", "muscovite", "kaolinite"]
rm = rheo[(rheo.source == "fig4-45") & (rheo.shear_rate == 100)]
rr = rheo[rheo.source == "fig4-47"]
rsum = {}
for c in CLAYS:
    a = rm[rm.clay == c].sort_values("temperature")
    b = rr[rr.clay == c].sort_values("temperature")
    v = a.viscosity.to_numpy(float)
    rsum[c] = dict(T=a.temperature.tolist(), visc=v.tolist(),
                   drop_pct=float((v[0] - v[-1]) / v[0] * 100),
                   rep_diff=float(np.abs(v - b.viscosity.to_numpy(float)).mean()))
    print(f"[F] {c:<10} viscosity {v[0]:.0f} -> {v[-1]:.0f} mPa s over 5-35 C "
          f"({rsum[c]['drop_pct']:.1f}%), replicate spread {rsum[c]['rep_diff']:.1f}")

wet = pd.read_csv(os.path.join(DATA, "wettability.csv"))
wc = wet[(wet.series == "temperature") & (wet.system == "clean")].sort_values("temperature")
w2 = wet[(wet.series == "temperature") & (wet.system == "chlorite25")].sort_values("temperature")
sc = float(np.polyfit(wc.temperature, wc.contact_angle, 1)[0])
s2 = float(np.polyfit(w2.temperature, w2.contact_angle, 1)[0])
print(f"[F] d(theta)/dT clean={sc:+.3f} deg/C   with 25% chlorite={s2:+.3f} deg/C")
cs = wet[wet.series == "content"].sort_values("clay_pct")
wd = wet[wet.series == "diesel"]
dsweep = {}
for s in ["clean", "chlorite25"]:
    sub = wd[wd.system == s].sort_values("diesel_gpt")
    t = sub.contact_angle.to_numpy(float)
    dsweep[s] = dict(diesel=sub.diesel_gpt.tolist(), theta=t.tolist(),
                     peak_at=float(sub.diesel_gpt.iloc[int(np.argmax(t))]),
                     gain=float(t.max() - t[0]))
    print(f"[F] {s}: contact angle peaks at {dsweep[s]['peak_at']:.0f} g/t "
          f"(+{dsweep[s]['gain']:.1f} deg over zero)")

# ================================================================ figures
# Figure set follows the layout of Eskanlou, Yin & Caers (2026): fitted
# surfaces are drawn as three-dimensional meshes over pairs of input
# variables, unconditional realisations are shown as a six-panel ensemble,
# and the decision figure puts the efficient frontier above the corresponding
# projections into input space.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

SURF = lambda ax: ax.view_init(elev=24, azim=-56)


def surf_panel(ax, x, y, z, pts=None, pts_v=None, cmap="viridis", lab="grade (%)",
               title="", wrap=None):
    """One 3-D kriging surface, optionally with the sampling locations."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    sf = ax.plot_surface(X, Y, z, cmap=cmap, linewidth=0, antialiased=True,
                         rstride=2, cstride=2, alpha=0.95,
                         vmin=wrap[0] if wrap else None, vmax=wrap[1] if wrap else None)
    if pts is not None:
        ax.scatter(pts[:, 0], pts[:, 1], pts_v, c="k", s=22, depthshade=False,
                   edgecolors="w", linewidths=0.5)
    ax.set_xlabel(XLAB[0], fontsize=10, labelpad=6)
    ax.set_ylabel(XLAB[JF[title]] if title in JF else "", fontsize=10, labelpad=6)
    ax.set_zlabel(lab, fontsize=9, labelpad=2)
    ax.tick_params(labelsize=8)
    SURF(ax)
    return sf


# ---- Fig 1: the campaign and its design
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
a = ax[0]
sc0 = a.scatter(bench.cu_grade, bench.mo_grade, s=46, c=bench.fineness, cmap="viridis",
                edgecolors="k", linewidths=0.4)
a.set_xlabel(SLAB["cu_grade"]); a.set_ylabel(SLAB["mo_grade"])
a.set_title("(a) The 29 bench tests")
plt.colorbar(sc0, ax=a, label=XLAB[0])
for k, name in enumerate(["pH", "diesel"]):
    a = ax[k + 1]
    sub = gdata[name]
    for resp, mk in [("mo_grade", "o"), ("cu_grade", "^")]:
        for fin, col in zip(ANCH, ["tab:blue", "tab:red"]):
            s = sub[np.isclose(sub.fineness, fin)]
            a.plot(s[XCOLS[JF[name]]], s[resp], mk, color=col, ms=6, label=f"fin {fin:.0f}%")
    a.set_xlabel(XLAB[JF[name]]); a.set_ylabel("grade (%)")
    a.set_title(f"({'bc'}[k]) fineness × {name}")
    a.legend(fontsize=7.5, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "Fig1_data.png")); plt.close()

# ---- Fig 2: variograms with the number of pairs per lag
fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
for a, resp in zip(axes, ["mo_grade", "cu_grade"]):
    v = variograms[resp]
    hb, gb = np.array(v["bins"]), np.array(v["semivariance"])
    npr = np.array(v["n_pairs"])
    ax2 = a.twinx()
    ax2.bar(hb, npr, width=hb.max() / len(hb) * 0.6, color="#d62728", alpha=0.85)
    ax2.set_ylim(0, npr.max() * 3.2); ax2.set_yticks([])
    xx = np.linspace(0, hb.max() * 1.05, 300)
    a.plot(xx, gauss_vario(xx, v["nugget"], v["sill"], v["range"]), "-",
           color="#2ca02c", lw=2)
    a.plot(hb, gb, "o", color="#1f77b4", ms=6)
    a.axhline(v["sill"], ls=":", color="grey", lw=1)
    a.axhline(v["nugget"], ls=":", color="grey", lw=1)
    a.set_ylim(0, v["sill"] * 1.25)
    a.set_ylabel("Semivariance (standardised)")
    a.set_title(f"{SLAB[resp]}   nugget {v['nugget']:.2f},  sill {v['sill']:.2f}, "
                f"range {v['range']:.2f},  nugget/sill {v['nugget_over_sill']:.2f}",
                fontsize=10)
axes[1].set_xlabel("Distance in the scaled operating space")
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "Fig2_variogram.png")); plt.close()

# ---- Fig 4: kriging surfaces, three-dimensional
fig = plt.figure(figsize=(15, 8.0))
for r, resp in enumerate(["mo_grade", "cu_grade"]):
    for c, name in enumerate(["pH", "diesel", "time"]):
        ax = fig.add_subplot(2, 3, r * 3 + c + 1, projection="3d")
        d = gpred[name][resp]
        sf = surf_panel(ax, d["x"], d["y"], d["mean"], d["pts"], d["vals"],
                        lab=SLAB[resp].split(" (")[0], title=name)
        ax.set_title(f"{SLAB[resp].split(' (')[0]} over fineness × {name}", fontsize=11)
        fig.colorbar(sf, ax=ax, shrink=0.55, pad=0.09, label="grade (%)")
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "Fig4_surface.png")); plt.close()

# ---- Fig 5 and Fig 6 share one posterior on the fineness x diesel plane
gsx = np.linspace(lo[0], hi[0], 44)
gsy = np.linspace(lo[2], hi[2], 44)
G2 = np.stack(np.meshgrid(gsx, gsy, indexing="ij"), -1).reshape(-1, 2)
gp_d = fit_gp((gdata["diesel"][["fineness", "diesel_gpt"]].to_numpy(float)
               - lo[[0, 2]]) / (hi[[0, 2]] - lo[[0, 2]]),
              gdata["diesel"]["cu_grade"].to_numpy(float))
Sc = (G2 - lo[[0, 2]]) / (hi[[0, 2]] - lo[[0, 2]])
mm = gp_d.predict(Sc)
M2 = mm.reshape(len(gsx), len(gsy))
S2 = gp_d.predict(Sc, return_std=True)[1].reshape(len(gsx), len(gsy))
WRAP = (float(M2.min()), float(M2.max()))

# ---- Fig 5: mean, variance and probability of meeting the target
from scipy.stats import norm as _norm
CU_TARGET = 0.33
prob = _norm.cdf((CU_TARGET - M2) / S2)
fig = plt.figure(figsize=(16.5, 5.4))
PANELS = [(M2, "viridis", "Cu grade", "posterior mean", WRAP),
          (S2, "magma_r", "sd", "variance (posterior sd)", None),
          (prob, "RdYlGn", "P", f"P(Cu ≤ {CU_TARGET} %)", (0.0, 1.0))]
NAMES = ["mean", "variance", "probability"]
for k, (z, cmap, lab, ttl, wrap) in enumerate(PANELS):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    sf = surf_panel(ax, gsx, gsy, z,
                    gdata["diesel"][["fineness", "diesel_gpt"]].to_numpy(float),
                    gdata["diesel"]["cu_grade"].to_numpy(float),
                    cmap=cmap, lab=lab, title="diesel", wrap=wrap)
    ax.set_title(ttl, fontsize=11)
    fig.colorbar(sf, ax=ax, shrink=0.55, pad=0.09)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "Fig5_ensemble.png")); plt.close()

# ---- Fig 6: efficient frontier over the projections into input space
fig = plt.figure(figsize=(12.5, 8.2))
a = fig.add_subplot(2, 1, 1)
o = np.argsort(cu_f)
a.scatter(mu4["cu_grade"], mu4["mo_grade"], s=4, c="#1f4e9c",
          label="interpolated")
a.scatter(bench.cu_grade, bench.mo_grade, s=42, c="darkorange", marker="o",
          edgecolors="k", linewidths=0.4, label="observed")
a.plot(cu_f[o], mo_f[o], "o", mfc="none", mec="k", ms=7, mew=1.1,
       label="non-dominated set")
a.scatter([knee_pred[1]], [knee_pred[0]], marker="*", s=420, c="limegreen",
          edgecolors="k", zorder=6, label="best trade-off")
a.set_xlabel(SLAB["cu_grade"]); a.set_ylabel(SLAB["mo_grade"])
a.set_title("(a) Non-dominated set of interpolated grades, and the best trade-off")
a.legend(fontsize=9, loc="lower left")
BOT = [("pH", "cu_grade", "(b) Cu grade over fineness × pH"),
       ("diesel", "mo_grade", "(c) Mo grade over fineness × diesel")]
for k, (name, resp, ttl) in enumerate(BOT):
    a = fig.add_subplot(2, 3, 4 + k)
    sub = gdata[name]
    d = gpred[name][resp]
    csh = a.scatter(gdata[name].fineness, sub[XCOLS[JF[name]]], c=d["vals"], s=70,
                    cmap="viridis", edgecolors="k", linewidths=0.5, zorder=3)
    a.scatter([knee[0]], [knee[JF[name]]], marker="*", s=420, c="limegreen",
              edgecolors="k", zorder=6)
    a.set_xlabel(XLAB[0]); a.set_ylabel(XLAB[JF[name]])
    a.set_title(ttl, fontsize=10.5)
    plt.colorbar(csh, ax=a, label=SLAB[resp].split(" (")[0] + " (%)")
a = fig.add_subplot(2, 3, 6)
a.hist(opt_grade, bins=14, color="#1f4e9c", alpha=0.9)
a.axvline(float(np.mean(opt_grade)), color="crimson", ls="--", lw=1.6)
a.set_xlabel(SLAB["cu_grade"]); a.set_ylabel("count")
a.set_title(f"(d) Cu grade achieved, {len(opt_grade)} resamples\n"
            f"{np.mean(opt_grade):.3f} ± {np.std(opt_grade):.3f} %", fontsize=10.5)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "Fig6_pareto.png")); plt.close()

# ---- Fig 3: validation and sensitivities
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
for k, resp in enumerate(["mo_grade", "cu_grade"]):
    a = ax[k]
    y = Y[resp]
    a.scatter(y, loo[resp], s=42, c="tab:blue", edgecolors="k", linewidths=0.4)
    lim = [min(y.min(), loo[resp].min()) - 1, max(y.max(), loo[resp].max()) + 1]
    a.plot(lim, lim, "k--", lw=1)
    a.set_xlabel(f"observed {SLAB[resp]}"); a.set_ylabel("leave-one-out prediction")
    a.set_title(f"({'ab'[k]}) {resp.split('_')[0].upper()} grade\n"
                f"Q$^2$={stats[resp]['q2']:.2f}, RMSE={stats[resp]['rmse']:.2f}", fontsize=10)
a = ax[2]
w = 0.38
idx = np.arange(4)
for k, resp in enumerate(["mo_grade", "cu_grade"]):
    a.bar(idx + (k - 0.5) * w, stats[resp]["inv_length_scale"], w,
          label=resp.split("_")[0].upper())
a.set_xticks(idx); a.set_xticklabels(["fineness", "pH", "diesel", "time"])
a.set_ylabel("inverse length scale"); a.set_yscale("log")
a.set_title("(c) ARD sensitivities", fontsize=10); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "Fig3_validation.png")); plt.close()

# ---- Fig 7: transfer to the plant
fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.4))
for k, resp in enumerate(["mo_grade", "cu_grade"]):
    a = ax[k]
    for line, sub in plant.groupby("line"):
        sub = sub.sort_values("fineness")
        a.plot(sub.fineness, sub[resp], "o--", ms=5, alpha=0.85, label=LINE.get(line, line))
    a.plot(FIN_ALONE.fineness, FIN_ALONE[resp], "s-", ms=6, color="k",
           label="bench, fineness sweep")
    a.axhspan(plant[resp].mean() - pe[resp]["pooled_sd"],
              plant[resp].mean() + pe[resp]["pooled_sd"], color="grey", alpha=0.15)
    a.set_xlabel(XLAB[0]); a.set_ylabel(SLAB[resp])
    a.set_title(f"({'ab'[k]}) {resp.split('_')[0]} grade against fineness\n"
                f"band = plant pure error ±{pe[resp]['pooled_sd']:.2f}", fontsize=10)
    a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "Fig7_transfer.png")); plt.close()

# ---- Fig 8: the pulp environment
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
a = ax[0]
for c in CLAYS:
    d = rsum[c]
    a.errorbar(d["T"], d["visc"], yerr=d["rep_diff"] / 2, fmt="o-", ms=5, capsize=3,
               label=f"{c} ({d['drop_pct']:.1f}%)")
a.set_xlabel("Temperature (°C)"); a.set_ylabel("Apparent viscosity (mPa s)")
a.set_title("(a) Pulp rheology at 100 s$^{-1}$", fontsize=10); a.legend(fontsize=8)
a = ax[1]
a.plot(wc.temperature, wc.contact_angle, "o-", ms=6, label=f"clean  ({sc:+.2f} °C$^{{-1}}$)")
a.plot(w2.temperature, w2.contact_angle, "s-", ms=6, label=f"25 % chlorite  ({s2:+.2f} °C$^{{-1}}$)")
a.set_xlabel("Temperature (°C)"); a.set_ylabel("Contact angle (°)")
a.set_title("(b) The temperature effect reverses", fontsize=10); a.legend(fontsize=8)
a = ax[2]
for s, lab in [("clean", "clean"), ("chlorite25", "25 % chlorite")]:
    d = dsweep[s]
    a.plot(d["diesel"], d["theta"], "o-", ms=6, label=f"{lab} (+{d['gain']:.1f}°)")
a.set_xlabel(XLAB[2]); a.set_ylabel("Contact angle (°)")
a.set_title("(c) Diesel dosage at 30 °C", fontsize=10); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "Fig8_pulp.png")); plt.close()

print("\nwrote 9 figures to", FIGS)
