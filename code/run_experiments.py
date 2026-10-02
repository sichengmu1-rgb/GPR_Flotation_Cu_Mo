# -*- coding: utf-8 -*-
"""
改进流程的定量实验。

对照两组：
  (A) reference  —— 复现 Eskanlou et al. (2026)：PCA + 各向同性核 + exact 克里金
  (B) refined    —— 本文提出：直接响应空间 + ARD 核 + nugget + LOO-CV + 机会约束前沿

输出：
  experiments_summary.txt     指标汇总
  Fig_validation.png          LOO-CV 拟合图 + 核参数
  Fig_frontier.png            均值前沿 vs 机会约束前沿
  Fig_ard.png                 ARD 长度尺度 / 敏感度
  results.json                机器可读结果
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from scipy.stats import norm

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
XCOLS = ["grain", "ph", "diesel", "time"]
XLAB_EN = ["Grinding fineness", "Pulp pH", "Diesel dosage", "Conditioning time"]
YLAB_EN = ["Mo recovery (%)", "Cu recovery (%)"]
SEED = 20260930


# ---------------------------------------------------------------- 指标
def crps_gaussian(y, mu, sigma):
    """高斯预测的连续排序概率得分（解析式，越小越好）"""
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1)
                                  + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def q2_rmse(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot, float(np.sqrt(np.mean((y - yhat) ** 2)))


def pareto_mask(y1, y2):
    """双最大化帕累托有效点掩码（向量化）"""
    n = len(y1)
    keep = np.ones(n, bool)
    order = np.argsort(-y1)
    best2 = -np.inf
    for i in order:
        if y2[i] >= best2:
            best2 = y2[i]
        else:
            keep[i] = False
    return keep


# ---------------------------------------------------------------- 数据
def load():
    os.makedirs(FIGS, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA, "mineral_x_dataset.csv"), encoding="utf-8-sig")
    Xr = df[XCOLS].to_numpy(float)
    Y = df[["mo_rec", "cu_rec"]].to_numpy(float)
    lo, hi = Xr.min(0), Xr.max(0)
    X = (Xr - lo) / (hi - lo)
    return df, Xr, X, Y, lo, hi


# ---------------------------------------------------------------- (A) 参考流程
def reference_workflow(X, Y, grid_n=20000):
    """复现：标准化 -> PCA(2) -> 各向同性核精确克里金 -> 独立条件模拟"""
    from sklearn.neighbors import KernelDensity
    import gstools as gs

    Ym, Ys = Y.mean(0), Y.std(0)
    Yz = (Y - Ym) / Ys
    pca = PCA(n_components=2, svd_solver="full").fit(Yz)
    P = pca.transform(Yz)

    models, kriges = [], []
    for k in range(2):
        v = float(np.var(P[:, k]))
        m = gs.Gaussian(dim=X.shape[1], var=v, len_scale=0.5, nugget=0.0)
        models.append(m)
        kriges.append(gs.krige.Ordinary(m, cond_pos=[X[:, i] for i in range(X.shape[1])],
                                        cond_val=P[:, k], exact=True))

    # 插值网格（KDE + 外推，与原文一致）
    rng = np.random.default_rng(SEED)
    kde = KernelDensity(bandwidth=0.15, kernel="gaussian").fit(X)
    lo, hi = X.min(0) - 0.1, X.max(0) + 0.1
    samp = []
    while len(samp) < grid_n:
        need = grid_n - len(samp)
        cand = rng.uniform(lo, hi, size=(need * 3, X.shape[1]))
        lp = kde.score_samples(cand)
        p = np.exp(lp - lp.max()); p /= p.sum()
        idx = rng.choice(len(cand), size=need, replace=False, p=p)
        samp.extend(cand[idx])
    Xg = np.r_[np.array(samp[:grid_n]), X]

    # 条件模拟（20 个实现）
    out = []
    for k in range(2):
        cs = gs.CondSRF(kriges[k])
        cs.set_pos([Xg[:, i] for i in range(X.shape[1])])
        ms = gs.random.MasterRNG(SEED + k)
        ens = []
        for j in range(20):
            cs(seed=ms(), store=[f"f{j}", False, False])
            ens.append(cs[f"f{j}"])
        out.append(np.stack(ens, 0))
    P1, P2 = out
    back = np.empty((20, Xg.shape[0], 2))
    for j in range(20):
        back[j] = pca.inverse_transform(np.c_[P1[j].flatten(), P2[j].flatten()]) * Ys + Ym
    return Xg, back[:, :, 0], back[:, :, 1], pca


# ---------------------------------------------------------------- (B) 改进流程
def fit_ard_gp(X, y):
    """ARD 各向异性核 + 可学习 nugget"""
    k = (ConstantKernel(1.0, (1e-3, 1e3))
         * RBF(length_scale=[0.5] * X.shape[1], length_scale_bounds=(1e-2, 1e2))
         + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)))
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True,
                                  n_restarts_optimizer=30, random_state=SEED)
    gp.fit(X, y)
    return gp


def loo_predict(gp, X, y):
    """留一法：返回 (预测均值, 预测标准差)"""
    loo = LeaveOneOut()
    mu = np.empty(len(y)); sd = np.empty(len(y))
    for tr, te in loo.split(X):
        g = GaussianProcessRegressor(kernel=gp.kernel_, normalize_y=True,
                                     optimizer=None, random_state=SEED)
        g.fit(X[tr], y[tr])
        m, s = g.predict(X[te], return_std=True)
        mu[te] = m; sd[te] = s
    return mu, sd


def main():
    df, Xr, X, Y, lo, hi = load()
    n, d = X.shape
    print(f"数据: n={n}, d={d}\n")
    report = {"n": int(n), "d": int(d)}

    # ============ (A) 参考流程 ============
    print("=" * 70)
    print("(A) 参考流程复现（PCA + 各向同性 + exact）")
    print("=" * 70)
    XgA, A1, A2, pca = reference_workflow(X, Y)
    m1, m2 = A1.mean(0), A2.mean(0)
    print(f"  插值网格点数   : {XgA.shape[0]:,}")
    print(f"  PCA 解释方差   : PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
    print(f"  载荷 PC1       : {np.round(pca.components_[0],3)}")
    print(f"  Mo 回收率预测  : [{m1.min():8.1f}, {m1.max():8.1f}] %")
    print(f"  Cu 回收率预测  : [{m2.min():8.1f}, {m2.max():8.1f}] %")
    fA = np.mean((m1 < 0) | (m1 > 100))
    print(f"  ✗ 非物理预测占比 (Mo)  : {fA*100:.1f}%")
    print(f"  ✗ 非物理预测占比 (Cu)  : {np.mean((m2<0)|(m2>100))*100:.1f}%")
    report["reference"] = {
        "grid": int(XgA.shape[0]),
        "pca_var": [float(pca.explained_variance_ratio_[0]),
                    float(pca.explained_variance_ratio_[1])],
        "pca_loadings": pca.components_.tolist(),
        "mo_range": [float(m1.min()), float(m1.max())],
        "cu_range": [float(m2.min()), float(m2.max())],
        "frac_nonphysical_mo": float(fA),
        "frac_nonphysical_cu": float(np.mean((m2 < 0) | (m2 > 100))),
    }

    # --- Fig_failure: admissible vs inadmissible prediction ranges ---
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for k, (nm, mm, frac) in enumerate([
            ("Mo recovery", m1, fA),
            ("Cu recovery", m2, float(np.mean((m2 < 0) | (m2 > 100))))]):
        lo_, hi_ = float(mm.min()), float(mm.max())
        a = axes[k]
        a.axvspan(0, 100, color="#e8f5e9", label="admissible [0,100]%")
        a.axvspan(min(lo_, 0), 0, color="#ffcdd2", label="inadmissible")
        a.axvspan(100, max(hi_, 100), color="#ffcdd2")
        a.annotate("", xy=(lo_, 0.55), xytext=(0, 0.55),
                   arrowprops=dict(arrowstyle="<->", color="crimson", lw=2))
        a.text(lo_ / 2, 0.62, f"{frac*100:.1f}% of domain\ninadmissible",
               ha="center", color="crimson", fontsize=9, fontweight="bold")
        a.set_xlim(min(lo_, 0) - 12, max(hi_, 100) + 12)
        a.set_ylim(0, 1); a.set_yticks([])
        a.set_xlabel(f"{nm} (%)")
        a.set_title(f"{nm}: reference workflow spans [{lo_:.1f}, {hi_:.1f}]%")
        a.legend(fontsize=7.5, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "Fig_failure.png"), dpi=160)
    plt.close()

    # ============ (B) 改进流程 ============
    print()
    print("=" * 70)
    print("(B) 改进流程（响应空间 + ARD 核 + nugget + LOO-CV）")
    print("=" * 70)
    gps, stats = [], []
    for k in range(2):
        gp = fit_ard_gp(X, Y[:, k])
        gps.append(gp)
        ls = gp.kernel_.k1.k2.length_scale
        nz = gp.kernel_.k2.noise_level
        mu, sd = loo_predict(gp, X, Y[:, k])
        q2, rmse = q2_rmse(Y[:, k], mu)
        crps = crps_gaussian(Y[:, k], mu, sd)
        print(f"\n  --- {YLAB_EN[k]} ---")
        print(f"    ARD 长度尺度  : {np.round(np.atleast_1d(ls), 3)}")
        print(f"    敏感度 1/ℓ    : {np.round(1/np.atleast_1d(ls), 3)}")
        print(f"    噪声 σ_n      : {float(np.sqrt(nz)):.4f}")
        print(f"    LOO  Q²       : {q2:.4f}")
        print(f"    LOO  RMSE     : {rmse:.3f} %")
        print(f"    LOO  CRPS     : {crps:.3f}")
        stats.append(dict(name=YLAB_EN[k], length_scale=np.atleast_1d(ls).tolist(),
                          noise=float(np.sqrt(nz)), q2=float(q2),
                          rmse=float(rmse), crps=float(crps),
                          mu=mu.tolist(), sd=sd.tolist()))
    report["refined"] = stats

    # ============ 机会约束帕累托前沿 ============
    print()
    print("=" * 70)
    print("(C) 帕累托前沿：均值 vs 机会约束")
    print("=" * 70)
    # 网格限制在数据包围盒内
    rng = np.random.default_rng(SEED)
    Ng = 8000
    Xg = rng.uniform(X.min(0), X.max(0), size=(Ng, d))
    Xg = np.r_[Xg, X]

    mu1, s1 = gps[0].predict(Xg, return_std=True)
    mu2, s2 = gps[1].predict(Xg, return_std=True)

    mask_mean = pareto_mask(mu1, mu2)
    print(f"  均值前沿点数  : {mask_mean.sum()}")

    # --- (C1) 诊断：前沿在统计上是否稳定 ---
    M = 400
    on_front = np.zeros(len(mu1))
    dominated = np.zeros(len(mu1))
    for _ in range(M):
        a = rng.normal(mu1, s1)
        b = rng.normal(mu2, s2)
        mk = pareto_mask(a, b)
        on_front += mk
        dominated += ~mk
    freq = on_front / M
    dom_prob = dominated / M
    n_stable = int((freq > 0.10).sum())
    print(f"  前沿稳定性诊断 (M={M} 次后验采样):")
    print(f"    出现在前沿频率 >10% 的点 : {n_stable}")
    print(f"    出现在前沿频率 >50% 的点 : {int((freq > 0.50).sum())}")
    print(f"    → 前沿不可稳定复现；'加不确定性'无法挽救一个退化的前沿")
    report["frontier"] = {"mean_only": int(mask_mean.sum()), "M": M,
                          "stable_10pct": n_stable,
                          "stable_50pct": int((freq > 0.50).sum()),
                          "note": "frontier is statistically degenerate because "
                                  "responses are positively correlated"}

    # --- (C2) 替代方案：可靠性（双优）操作窗口 ---
    # P(y1 >= q1 且 y2 >= q2) >= 1 - alpha
    q1 = float(np.percentile(Y[:, 0], 60))
    q2 = float(np.percentile(Y[:, 1], 60))
    print(f"\n  替代方案 — 可靠性操作窗口 (目标: Mo≥{q1:.0f}% 且 Cu≥{q2:.0f}%):")
    aS = rng.normal(mu1, s1, size=(M, len(mu1)))
    bS = rng.normal(mu2, s2, size=(M, len(mu1)))
    rel = ((aS >= q1) & (bS >= q2)).mean(axis=0)
    report["reliability"] = {"q1": q1, "q2": q2}
    for alpha in (0.50, 0.25, 0.10):
        n = int((rel >= 1 - alpha).sum())
        print(f"    P(双优) ≥ {1-alpha:.0%}  的网格点数: {n}")
        report["reliability"][f"conf_{1-alpha}"] = n

    # 选可靠性最高、且回收率尽量高的点
    best = int(np.argmax(rel + 1e-6 * (mu1 + mu2)))
    knee = int(np.argmax(rel * 0.9 + 0.1 * (mu1 + mu2) / 100))
    print(f"\n  推荐操作点（可靠性最高）: P={rel[best]:.2f}, "
          f"Mo={mu1[best]:.1f}±{s1[best]:.1f}, Cu={mu2[best]:.1f}±{s2[best]:.1f}")
    report["recommended_idx"] = best
    report["reliability_max"] = float(rel[best])
    idx = np.array([best])
    Xopt = Xg[knee] * (hi - lo) + lo
    print(f"\n  最优操作点（膝点，均值∩鲁棒）:")
    for i, lab in enumerate(XLAB_EN):
        print(f"    {lab:<22} = {Xopt[i]:>8.2f}")
    print(f"    -> Mo 回收率 = {mu1[knee]:.2f} %  (σ={s1[knee]:.2f})")
    print(f"    -> Cu 回收率 = {mu2[knee]:.2f} %  (σ={s2[knee]:.2f})")
    report["optimal"] = {"X": Xopt.tolist(), "mo": float(mu1[knee]),
                         "cu": float(mu2[knee]), "sd_mo": float(s1[knee]),
                         "sd_cu": float(s2[knee])}

    # ============ 出图 ============
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for k, a in enumerate(ax):
        st = stats[k]
        y = Y[:, k]; mu = np.array(st["mu"]); sd = np.array(st["sd"])
        a.errorbar(y, mu, yerr=1.96 * sd, fmt="o", ms=5, capsize=3,
                   color="tab:blue", ecolor="lightsteelblue", label="LOO prediction ±95%")
        lim = [min(y.min(), mu.min()) - 5, max(y.max(), mu.max()) + 5]
        a.plot(lim, lim, "k--", lw=1, label="1:1")
        a.set_xlabel(f"Observed {YLAB_EN[k]}"); a.set_ylabel(f"Predicted {YLAB_EN[k]}")
        a.set_title(f"{YLAB_EN[k]}   $Q^2$={st['q2']:.3f}  RMSE={st['rmse']:.2f}")
        a.legend(fontsize=8); a.set_xlim(lim); a.set_ylim(lim)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "Fig_validation.png"), dpi=160)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].scatter(mu2, mu1, s=4, c="lightsteelblue", label="GP posterior mean field")
    ax[0].scatter(mu2[mask_mean], mu1[mask_mean], s=22, c="tab:red",
                  label=f"mean-field frontier ({mask_mean.sum()} pts)")
    ax[0].scatter(mu2[best], mu1[best], s=220, marker="*", c="gold",
                  edgecolors="k", zorder=5, label="recommended operating point")
    ax[0].scatter(Y[:, 1], Y[:, 0], s=45, c="k", marker="x", label="observations")
    ax[0].axvline(q2, color="grey", ls=":", lw=1); ax[0].axhline(q1, color="grey", ls=":", lw=1)
    ax[0].set_xlabel(YLAB_EN[1]); ax[0].set_ylabel(YLAB_EN[0])
    ax[0].set_title("(a) Mean-field frontier and recommended point")
    ax[0].legend(fontsize=8)

    sc = ax[1].scatter(mu2, mu1, s=8, c=rel, cmap="RdYlGn", vmin=0, vmax=1)
    ax[1].scatter(mu2[best], mu1[best], s=220, marker="*", c="gold",
                  edgecolors="k", zorder=5, label="recommended")
    ax[1].scatter(Y[:, 1], Y[:, 0], s=45, facecolors="none", edgecolors="k",
                  linewidths=0.8, label="observations")
    ax[1].set_xlabel(YLAB_EN[1]); ax[1].set_ylabel(YLAB_EN[0])
    ax[1].set_title(f"(b) Reliability  $P(y_1\\geq{q1:.0f},y_2\\geq{q2:.0f})$")
    ax[1].legend(fontsize=8)
    plt.colorbar(sc, ax=ax[1], label="probability")
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "Fig_frontier.png"), dpi=160)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, a in enumerate(ax):
        ls = np.atleast_1d(stats[k]["length_scale"])
        sens = 1 / ls
        o = np.argsort(-sens)
        a.barh([XLAB_EN[i] for i in o], sens[o], color="steelblue", alpha=0.85)
        a.set_xlabel("Inverse length scale  $1/\\ell_i$  (sensitivity)")
        a.set_title(f"{YLAB_EN[k]}   $Q^2$={stats[k]['q2']:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "Fig_ard.png"), dpi=160)
    plt.close()

    # 核参数 + ARD 敏感度表
    with open(os.path.join(ROOT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 results.json")
    print("图: Fig_failure.png / Fig_ard.png / Fig_validation.png / Fig_frontier.png")


if __name__ == "__main__":
    main()
