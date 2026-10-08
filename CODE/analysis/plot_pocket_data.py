#!/usr/bin/env python3
"""
plot_pocket_data.py — 可视化 p2rank_pocket_data.csv 的 pocket 级分布。

输出：
  results/figs/fig_pocket_rank_dist.png    正确 pocket 在排序中的位置分布
  results/figs/fig_score_vs_dist.png       P2Rank score 与真值距离关系
  results/figs/fig_energy_vs_dist.png      Vina energy 与真值距离关系
  results/figs/fig_top1_vs_best.png        每个 case 的 top1 误差 vs 最佳误差
  results/figs/fig_score_energy_joint.png  P2Rank score vs Vina energy（按距离着色）
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["font.weight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "bold"
matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path("results/figs")
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300


def load_data(path):
    by_case = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                by_case[r["case"]].append({
                    "rank": int(r["rank"]),
                    "p2rank_score": float(r["p2rank_score"]),
                    "vina_energy": float(r["vina_energy"]) if r["vina_energy"] else None,
                    "err_to_true": float(r["err_to_true"]),
                })
            except (ValueError, KeyError):
                continue
    for c in by_case:
        by_case[c].sort(key=lambda p: p["rank"])
    return by_case


# ============================================================
# Fig 1：正确 pocket 在 P2Rank / Vina 排序中的位置
# ============================================================
def fig_rank_dist(by_case):
    prank_positions, vrank_positions = [], []

    for pockets in by_case.values():
        correct = [p for p in pockets if p["err_to_true"] < 4.0]
        if not correct:
            continue

        pr = min(p["rank"] for p in correct)
        prank_positions.append(pr)

        ve = [p for p in pockets if p["vina_energy"] is not None]
        ve_sorted = sorted(ve, key=lambda p: p["vina_energy"])
        for i, p in enumerate(ve_sorted, 1):
            if p["err_to_true"] < 4.0:
                vrank_positions.append(i)
                break

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0.5, 21.5, 1)

    ax.hist(prank_positions, bins=bins, alpha=0.6,
            color="#1f77b4", label=f"P2Rank (n={len(prank_positions)})",
            edgecolor="k", linewidth=0.5)
    ax.hist(vrank_positions, bins=bins, alpha=0.6,
            color="#d62728", label=f"Vina (n={len(vrank_positions)})",
            edgecolor="k", linewidth=0.5)

    ax.set_xlabel("Rank of the best correct pocket (< 4 Å)", fontsize=13)
    ax.set_ylabel("Number of cases", fontsize=13)
    ax.set_title("Where is the correct pocket in the ranking?",
                 fontsize=13, pad=35)
    # 图注放到标题下方的中间
    ax.legend(fontsize=11, frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)

    fig.tight_layout()
    p = OUT / "fig_pocket_rank_dist.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {p}")


# ============================================================
# Fig 2：P2Rank score vs 距离真值
# ============================================================
def fig_score_vs_dist(by_case):
    scores, dists, is_top1 = [], [], []
    for pockets in by_case.values():
        for p in pockets:
            scores.append(p["p2rank_score"])
            dists.append(p["err_to_true"])
            is_top1.append(p["rank"] == 1)
    scores = np.array(scores); dists = np.array(dists)
    is_top1 = np.array(is_top1)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(scores[~is_top1], dists[~is_top1],
               s=6, alpha=0.25, c="#1f77b4", edgecolors="none",
               label="Pocket rank > 1")
    ax.scatter(scores[is_top1], dists[is_top1],
               s=16, alpha=0.6, c="#d62728", edgecolors="none",
               label="Pocket rank = 1 (top1)")
    ax.axhline(4, color="k", ls="--", lw=0.8)

    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(scores, dists)
        ax.set_title(f"P2Rank score vs distance to true center\n"
                     f"Spearman ρ = {rho:+.3f}, p = {pval:.1e}",
                     fontsize=12, pad=45)
    except ImportError:
        ax.set_title("P2Rank score vs distance to true center",
                     fontsize=12, pad=35)

    ax.set_xlabel("P2Rank score", fontsize=13)
    ax.set_ylabel("Distance to true center (Å)", fontsize=13)
    # 图注放到标题下方的中间
    ax.legend(fontsize=10, frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(True, alpha=0.25, lw=0.4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)

    fig.tight_layout()
    p = OUT / "fig_score_vs_dist.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {p}")


# ============================================================
# Fig 3：Vina energy vs 距离真值
# ============================================================
def fig_energy_vs_dist(by_case):
    energies, dists, is_top1 = [], [], []
    for pockets in by_case.values():
        for p in pockets:
            if p["vina_energy"] is None:
                continue
            energies.append(p["vina_energy"])
            dists.append(p["err_to_true"])
            is_top1.append(p["rank"] == 1)
    energies = np.array(energies); dists = np.array(dists)
    is_top1 = np.array(is_top1)

    if len(energies) < 5:
        print("  [WARN] Vina energy 数据不足，跳过")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(energies[~is_top1], dists[~is_top1],
               s=6, alpha=0.25, c="#1f77b4", edgecolors="none",
               label="Pocket rank > 1")
    ax.scatter(energies[is_top1], dists[is_top1],
               s=16, alpha=0.6, c="#d62728", edgecolors="none",
               label="Pocket rank = 1")
    ax.axhline(4, color="k", ls="--", lw=0.8)

    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(energies, dists)
        ax.set_title(f"Vina energy vs distance to true center\n"
                     f"Spearman ρ = {rho:+.3f}, p = {pval:.1e}",
                     fontsize=12, pad=45)
    except ImportError:
        ax.set_title("Vina energy vs distance to true center",
                     fontsize=12, pad=35)

    ax.set_xlabel("Vina energy (kcal/mol)", fontsize=13)
    ax.set_ylabel("Distance to true center (Å)", fontsize=13)
    # 图注放到标题下方的中间
    ax.legend(fontsize=10, frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(True, alpha=0.25, lw=0.4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)

    fig.tight_layout()
    p = OUT / "fig_energy_vs_dist.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {p}")


# ============================================================
# Fig 4：每个 case 的 top1 误差 vs 最佳误差
# ============================================================
def fig_top1_vs_best(by_case):
    top1, best = [], []
    for pockets in by_case.values():
        if not pockets:
            continue
        top1.append(pockets[0]["err_to_true"])
        best.append(min(p["err_to_true"] for p in pockets))
    top1 = np.array(top1); best = np.array(best)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(top1, best, s=10, alpha=0.5, c="#1f77b4",
               edgecolors="none")
    lim = max(top1.max(), best.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
    ax.axhline(4, color="#d62728", ls=":", lw=1.0)
    ax.axvline(4, color="#d62728", ls=":", lw=1.0)

    ax.set_xlabel("Top1 pocket error (Å)", fontsize=13)
    ax.set_ylabel("Best pocket error (Å) within top 20", fontsize=13)
    ax.set_title(f"Top1 vs best-in-top20 error (n = {len(top1)})",
                 fontsize=13, pad=35)
    # 图注放到标题下方的中间
    ax.legend(fontsize=11, frameon=False, ncol=1,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.25, lw=0.4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)

    fig.tight_layout()
    p = OUT / "fig_top1_vs_best.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {p}")


# ============================================================
# Fig 5：P2Rank score vs Vina energy，按距离着色（用 colorbar，无 legend）
# ============================================================
def fig_joint(by_case):
    s, e, d = [], [], []
    for pockets in by_case.values():
        for p in pockets:
            if p["vina_energy"] is None:
                continue
            s.append(p["p2rank_score"])
            e.append(p["vina_energy"])
            d.append(p["err_to_true"])
    s = np.array(s); e = np.array(e); d = np.array(d)

    if len(s) < 5:
        print("  [WARN] 数据不足，跳过")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    d_clip = np.clip(d, 0, 20)
    sc = ax.scatter(s, e, c=d_clip, cmap="RdYlGn_r",
                    s=8, alpha=0.6, edgecolors="none",
                    vmin=0, vmax=20)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Distance to true center (Å)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("P2Rank score", fontsize=13)
    ax.set_ylabel("Vina energy (kcal/mol)", fontsize=13)
    ax.set_title("Joint distribution of P2Rank score and Vina energy\n"
                 "(colored by distance to true center)",
                 fontsize=12, pad=15)
    ax.grid(True, alpha=0.25, lw=0.4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)

    fig.tight_layout()
    p = OUT / "fig_score_energy_joint.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {p}")


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket_csv", default="results/p2rank_pocket_data.csv")
    args = ap.parse_args()

    print(f"加载 {args.pocket_csv} ...")
    by_case = load_data(args.pocket_csv)
    print(f"  case 数: {len(by_case)}")
    total = sum(len(v) for v in by_case.values())
    print(f"  pocket 总数: {total}")

    print("\n画图:")
    fig_rank_dist(by_case)
    fig_score_vs_dist(by_case)
    fig_energy_vs_dist(by_case)
    fig_top1_vs_best(by_case)
    fig_joint(by_case)

    print(f"\n完成，图在 {OUT.resolve()}")


if __name__ == "__main__":
    main()