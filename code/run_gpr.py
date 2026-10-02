# -*- coding: utf-8 -*-
"""
复现 Eskanlou, Yin & Caers (2026) Minerals Engineering 237:110000
"Gaussian process regression for modeling computational and experimental
 mineral processing data" 的实验数据工作流（Notebook 02）。

把该流程套用到宋学文的铜钼浮选分离试验数据上。

输入: mineral_x_dataset.csv   (X: 细度/pH/柴油/时间, Y: Mo回收率/Cu回收率)
输出: 帕累托前沿 + 最优工艺参数 + 图
"""

import sys
import os
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import gstools as gs

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False

SEED = 20170519
ENS = 20
N_GRID = 20000

XCOLS = ["grain", "ph", "diesel", "time"]
XLAB = ["细度(-0.074mm,%)", "矿浆 pH", "柴油用量(g/t)", "作用时间(min)"]
YLAB = ["Mo 回收率 (%)", "Cu 回收率 (%)"]


def main():
    os.makedirs(FIGS, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA, "mineral_x_dataset.csv"), encoding="utf-8-sig")
    print(f"数据: {len(df)} 个试验点\n")

    X_raw = df[XCOLS].to_numpy(dtype=float)
    Y = df[["mo_rec", "cu_rec"]].to_numpy(dtype=float)

    # ---- 与文中一致: 预测变量 min-max 缩放到 [0,1] ----
    xmin, xmax = X_raw.min(axis=0), X_raw.max(axis=0)
    X = (X_raw - xmin) / (xmax - xmin)
    print("X 缩放范围:")
    for i, c in enumerate(XCOLS):
        print(f"  {c:<6} 原始 [{xmin[i]:>7.2f}, {xmax[i]:>7.2f}]")
    print()

    # ---- 1. PCA on standardized Y ----
    Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)
    Y_stdrd = (Y - Y_mean) / Y_std
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(Y_stdrd)
    PCs = pca.transform(Y_stdrd)
    PC1, PC2 = PCs[:, 0], PCs[:, 1]
    ev = pca.explained_variance_ratio_
    print(f"PCA 解释方差: PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%")
    print(f"  载荷 PC1 = {np.round(pca.components_[0], 3)}   (Mo, Cu)")
    print(f"  载荷 PC2 = {np.round(pca.components_[1], 3)}")
    print()

    # ---- 2. 经验变差函数 + 拟合 ----
    print("变差函数拟合:")
    import skgstat as skg
    vario = {}
    for name, val in (("PC1", PC1), ("PC2", PC2)):
        V = skg.Variogram(X, val, model="gaussian", maxlag=0.9, n_lags=9, normalize=False)
        try:
            V.fit_method = "trf"
            V.fit()
            r, s, n = V.parameters
        except Exception as e:
            r, s, n = 0.5, float(np.var(val)), 0.05
        vario[name] = (r, s, n)
        print(f"  {name}: range={r:.3f}  sill={s:.3f}  nugget={n:.3f}")
    print()

    # ---- 3. 高斯模型 + 普通克里金 ----
    r1, s1, n1 = vario["PC1"]
    r2, s2, n2 = vario["PC2"]
    model_pc1 = gs.Gaussian(dim=X.shape[1], var=max(s1, 1e-3), len_scale=max(r1, 1e-2), nugget=max(n1, 0))
    model_pc2 = gs.Gaussian(dim=X.shape[1], var=max(s2, 1e-3), len_scale=max(r2, 1e-2), nugget=max(n2, 0))

    coords = [X[:, i] for i in range(X.shape[1])]
    ok_pc1 = gs.krige.Ordinary(model_pc1, cond_pos=coords, cond_val=PC1, exact=True)
    ok_pc2 = gs.krige.Ordinary(model_pc2, cond_pos=coords, cond_val=PC2, exact=True)
    print("普通克里金构建完成")

    # ---- 4. 生成插值网格（KDE 密度约束）----
    def sample_near_X(Xs, n_samples, bandwidth=0.15, seed=SEED):
        """KDE 密度约束采样；坐标严格限制在数据包围盒内（不外推）。
        原文用的是 min-0.1 / max+0.1，但我们的数据只有 38 点且维度分布不均，
        外推会产生负回收率，所以收紧到数据范围内。"""
        np.random.seed(seed)
        N, D = Xs.shape
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(Xs)
        lo = np.maximum(Xs.min(axis=0) - 0.02, 0.0)
        hi = np.minimum(Xs.max(axis=0) + 0.02, 1.0)
        out = []
        while len(out) < n_samples:
            need = n_samples - len(out)
            cand = np.random.uniform(lo, hi, size=(need * 3, D))
            lp = kde.score_samples(cand)
            p = np.exp(lp - lp.max()); p /= p.sum()
            idx = np.random.choice(len(cand), size=need, replace=False, p=p)
            out.extend(cand[idx])
        return np.r_[np.array(out[:n_samples]), Xs]

    X_grid = sample_near_X(X, N_GRID, 0.15)
    print(f"插值网格: {X_grid.shape[0]} 点（范围限制在数据包围盒内）\n")

    # ---- 5. 条件随机场 (GPR) ----
    def cond_srf(ok, arr):
        c = gs.CondSRF(ok)
        c.set_pos([X_grid[:, i] for i in range(X.shape[1])])
        return c

    csrf1, csrf2 = cond_srf(ok_pc1, PC1), cond_srf(ok_pc2, PC2)

    seed = gs.random.MasterRNG(SEED)
    f1 = {i: None for i in range(ENS)}
    f2 = {i: None for i in range(ENS)}
    print(f"条件模拟 ({ENS} 个实现)...")
    for i in range(ENS):
        csrf1(seed=seed(), store=[f"f{i}", False, False])
        f1[i] = csrf1[f"f{i}"]
    for i in range(ENS):
        csrf2(seed=seed(), store=[f"g{i}", False, False])
        f2[i] = csrf2[f"g{i}"]
    print("完成\n")

    # ---- 6. 反变换 PC -> Y ----
    back = []
    for i in range(ENS):
        pcs = np.c_[f1[i].flatten(), f2[i].flatten()]
        back.append(pca.inverse_transform(pcs) * Y_std + Y_mean)
    back = np.asarray(back)          # (ens, n_grid, 2)

    # 集合均值作为预测；回收率物理上限定在 [0, 100]
    Y1_pred = np.clip(back[:, :, 0].mean(axis=0), 0.0, 100.0)
    Y2_pred = np.clip(back[:, :, 1].mean(axis=0), 0.0, 100.0)
    Y1_std = back[:, :, 0].std(axis=0)
    Y2_std = back[:, :, 1].std(axis=0)
    print(f"预测 Mo 回收率: {Y1_pred.min():.1f} ~ {Y1_pred.max():.1f} %")
    print(f"预测 Cu 回收率: {Y2_pred.min():.1f} ~ {Y2_pred.max():.1f} %")
    print(f"Mo 不确定度(σ) 平均: {Y1_std.mean():.2f}")
    print()

    # ---- 7. 帕累托前沿 ----
    def get_efficient_frontier(Yin):
        n = Yin.shape[0]
        eff = np.ones(n, dtype=bool)
        for i in range(n):
            if eff[i]:
                eff[eff] = np.any(Yin[eff] > Yin[i], axis=1) | np.all(Yin[eff] == Yin[i], axis=1)
                eff[i] = True
        return Yin[eff], eff, np.where(eff)[0]

    Y_grid = np.c_[Y1_pred, Y2_pred]
    front, mask, idx = get_efficient_frontier(Y_grid)
    print(f"帕累托前沿: {len(idx)} 个点\n")

    # 选"拐点" —— 用到理想点(最大Mo,最大Cu)的归一化距离最小
    y1n = (front[:, 0] - front[:, 0].min()) / (np.ptp(front[:, 0]) + 1e-12)
    y2n = (front[:, 1] - front[:, 1].min()) / (np.ptp(front[:, 1]) + 1e-12)
    d = np.sqrt((1 - y1n) ** 2 + (1 - y2n) ** 2)
    best_local = int(np.argmin(d))
    best_global = int(idx[best_local])

    X_best_scaled = np.array([X_grid[:, k][best_global] for k in range(X.shape[1])])
    X_best = X_best_scaled * (xmax - xmin) + xmin

    print("=" * 62)
    print("最优工艺参数（帕累托前沿拐点）")
    print("=" * 62)
    for i, lab in enumerate(XLAB):
        print(f"  {lab:<22} = {X_best[i]:>8.2f}")
    print(f"\n  对应预测: Mo 回收率 = {Y1_pred[best_global]:.2f} %")
    print(f"            Cu 回收率 = {Y2_pred[best_global]:.2f} %")
    print(f"            Mo 品位取向的不确定度 σ = {Y1_std[best_global]:.2f} %")

    # ---- 8. 出图 ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    sc = ax.scatter(Y2_pred, Y1_pred, s=2, c=Y1_std, cmap="viridis", alpha=0.6)
    ax.scatter(Y2_pred[idx], Y1_pred[idx],
               s=90, facecolors="none", edgecolors="r", linewidths=1.4, label="帕累托前沿")
    ax.scatter(Y2_pred[best_global], Y1_pred[best_global], s=180, marker="*",
               c="gold", edgecolors="k", zorder=5, label="最优权衡点")
    ax.scatter(Y[:, 1], Y[:, 0], s=45, c="k", marker="x", label="实测点")
    ax.set_xlabel(YLAB[1]); ax.set_ylabel(YLAB[0])
    ax.set_title("(a) Mo vs Cu 回收率 与帕累托前沿")
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="Mo 预测 σ (%)")

    ax = axes[1]
    fo = front[np.argsort(front[:, 1])]
    ax.plot(fo[:, 1], fo[:, 0], "r-o", ms=4, lw=1.2)
    ax.scatter(Y[:, 1], Y[:, 0], s=45, c="k", marker="x", label="实测点")
    ax.scatter(front[best_local, 1], front[best_local, 0], s=180, marker="*",
               c="gold", edgecolors="k", zorder=5, label="最优权衡点")
    ax.set_xlabel(YLAB[1]); ax.set_ylabel(YLAB[0])
    ax.set_title("(b) 前沿放大")
    ax.legend(fontsize=8)

    ax = axes[2]
    names = ["细度", "pH", "柴油", "时间"]
    ax.barh(names, X_best_scaled, color="steelblue", alpha=0.85)
    ax.set_xlim(0, 1); ax.set_xlabel("归一化值 [0,1]")
    ax.set_title("(c) 最优工艺参数")
    for i, v in enumerate(X_best_scaled):
        ax.text(v + 0.02, i, f"{X_best[i]:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(FIGS, "GPR_result.png")
    plt.savefig(out_png, dpi=150)
    print(f"\n图已保存: {out_png}")

    # 结果表
    res = pd.DataFrame({
        "X1_细度": [X_best[0]], "X2_pH": [X_best[1]], "X3_柴油": [X_best[2]], "X4_时间": [X_best[3]],
        "Mo回收率预测": [Y1_pred[best_global]], "Cu回收率预测": [Y2_pred[best_global]],
    })
    res.to_csv(os.path.join(ROOT, "GPR_optimal.csv"), index=False, encoding="utf-8-sig")
    front_df = pd.DataFrame(fo, columns=["Mo回收率", "Cu回收率"])
    front_df.to_csv(os.path.join(ROOT, "GPR_pareto_front.csv"), index=False, encoding="utf-8-sig")
    print("最优参数表: GPR_optimal.csv")
    print("帕累托前沿: GPR_pareto_front.csv")


if __name__ == "__main__":
    main()
