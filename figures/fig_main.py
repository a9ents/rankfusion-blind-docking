#!/usr/bin/env python3
"""
make_figures.py — 生成论文用图（修正版）

读取：
  results/sidechain_rmsd.csv
  results/af3_analysis_bins.csv

输出（results/figs/ 下）：
  fig1_backbone_vs_err.png       主链 RMSD vs 盒子误差（散点）
  fig2_sidechain_vs_err.png      侧链 RMSD vs 盒子误差（散点）
  fig3_bins_hit_rate.png         三种 RMSD 分档成功率（柱状）
  fig4_iptm_vs_hit.png           ipTM 分档成功率（柱状）
  fig5_matched_vs_unmatched.png  配对 vs 未配对成功率（柱状）
  fig6_global_rmsd_vs_err.png    全局 RMSD vs 盒子误差（散点，对比）
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 字体：优先中文，没有就用默认
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"
]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path("results/figs")
OUT.mkdir(parents=True, exist_ok=True)
DPI = 200


def read_csv(path):
    if not Path(path).is_file():
        print(f"  ⚠ 缺文件: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fl(v, default=None):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# =====================================================================
# 散点图
# =====================================================================
def scatter_plot(rows, xkey, xlabel, out_name, xcut=None, xcut_label=None):
    xs, ys, hit = [], [], []
    for r in rows:
        x = fl(r.get(xkey))
        y = fl(r.get("box_err_A"))
        h = fl(r.get("hit_4A"))
        if x is None or y is None or h is None:
            continue
        xs.append(x); ys.append(y); hit.append(int(h))

    if not xs:
        print(f"  ⚠ {out_name}: 无数据")
        return

    xs = np.array(xs); ys = np.array(ys); hit = np.array(hit)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs[hit == 0], ys[hit == 0], s=9, c="#999999",
               alpha=0.5, label="miss")
    ax.scatter(xs[hit == 1], ys[hit == 1], s=22, c="#d62728",
               alpha=0.85, edgecolors="k", linewidths=0.3,
               label="hit (<4 A)")
    ax.axhline(4, color="k", linestyle="--", lw=0.8,
               label="4 A threshold")
    if xcut is not None:
        ax.axvline(xcut, color="#1f77b4", linestyle=":", lw=0.9,
                   label=xcut_label or f"{xcut} A cutoff")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Box center error (A)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25, lw=0.4)
    fig.tight_layout()
    p = OUT / out_name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"  ✅ {p}")


# =====================================================================
# 三种 RMSD 分档成功率
# =====================================================================
def bars_three_features(bins_rows, out_name):
    feats = ["sidechain_rmsd", "backbone_rmsd", "global_rmsd"]
    titles = ["Sidechain RMSD", "Backbone RMSD (pocket)", "Global RMSD"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    data = {f: [] for f in feats}
    for r in bins_rows:
        if r.get("feature") in data:
            data[r["feature"]].append(r)

    if not any(data.values()):
        print(f"  ⚠ {out_name}: 无数据")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, f, title, c in zip(axes, feats, titles, colors):
        rs = data[f]
        if not rs:
            ax.set_title(title)
            continue

        labels = [r["bin"] for r in rs]
        # 注意：af3_analysis_bins.csv 的列名是 hit_rate_4A
        rates = [float(r["hit_rate_4A"]) * 100 for r in rs]
        ns = [int(r["n"]) for r in rs]

        bars = ax.bar(range(len(labels)), rates, color=c, alpha=0.75,
                      edgecolor="k", linewidth=0.4)
        for b, n in zip(bars, ns):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                    f"n={n}", ha="center", fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, lw=0.4)
    axes[0].set_ylabel("Success rate (<4 A, %)", fontsize=11)

    fig.tight_layout()
    p = OUT / out_name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"  ✅ {p}")


# =====================================================================
# ipTM 分档成功率
# =====================================================================
def bar_iptm(bins_rows, out_name):
    rs = [r for r in bins_rows if r.get("feature") == "ipTM"]
    if not rs:
        print(f"  ⚠ {out_name}: 无 ipTM 数据")
        return

    labels = [r["bin"] for r in rs]
    rates = [float(r["hit_rate_4A"]) * 100 for r in rs]
    ns = [int(r["n"]) for r in rs]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(range(len(labels)), rates, color="#9467bd",
                  alpha=0.78, edgecolor="k", linewidth=0.4)
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                f"n={n}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_xlabel("AF3 ipTM", fontsize=12)
    ax.set_ylabel("Success rate (<4 A, %)", fontsize=12)
    ax.set_title("Rank Fusion success vs AF3 confidence", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout()
    p = OUT / out_name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"  ✅ {p}")


# =====================================================================
# 配对 vs 未配对
# =====================================================================
def bar_matched(rows, out_name):
    matched_hits, unmatched_hits = [], []
    for r in rows:
        h = fl(r.get("hit_4A"))
        if h is None:
            continue
        sc = r.get("sidechain_rmsd_A", "")
        if sc not in ("", None):
            matched_hits.append(int(h))
        else:
            unmatched_hits.append(int(h))

    if not matched_hits and not unmatched_hits:
        print(f"  ⚠ {out_name}: 无数据")
        return

    labels = ["Matched", "Unmatched"]
    rates = [np.mean(matched_hits) * 100 if matched_hits else 0,
             np.mean(unmatched_hits) * 100 if unmatched_hits else 0]
    ns = [len(matched_hits), len(unmatched_hits)]

    fig, ax = plt.subplots(figsize=(5, 4.2))
    bars = ax.bar(labels, rates, color=["#2ca02c", "#d62728"],
                  alpha=0.78, edgecolor="k", linewidth=0.5)
    for b, n, rate in zip(bars, ns, rates):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                f"n={n}\n{rate:.2f}%", ha="center", fontsize=10)
    ax.set_ylabel("Success rate (<4 A, %)", fontsize=12)
    ax.set_title("Pocket residue matching vs success", fontsize=11)
    ax.set_ylim(0, max(rates) * 1.35 + 1)
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout()
    p = OUT / out_name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"  ✅ {p}")


# =====================================================================
# 主流程
# =====================================================================
def main():
    print("生成论文图 ...\n")

    sc_rows = read_csv("results/sidechain_rmsd.csv")
    bin_rows = read_csv("results/af3_analysis_bins.csv")

    if sc_rows:
        scatter_plot(sc_rows, "backbone_rmsd_A",
                     "Pocket backbone RMSD (A)",
                     "fig1_backbone_vs_err.png",
                     xcut=3.0, xcut_label="3 A cutoff")
        scatter_plot(sc_rows, "sidechain_rmsd_A",
                     "Pocket sidechain RMSD (A)",
                     "fig2_sidechain_vs_err.png")
        scatter_plot(sc_rows, "global_rmsd_A",
                     "Global RMSD (A)",
                     "fig6_global_rmsd_vs_err.png",
                     xcut=8.0, xcut_label="8 A cutoff")
        bar_matched(sc_rows, "fig5_matched_vs_unmatched.png")

    if bin_rows:
        bars_three_features(bin_rows, "fig3_bins_hit_rate.png")
        bar_iptm(bin_rows, "fig4_iptm_vs_hit.png")

    print(f"\n完成，图在 {OUT}/")


if __name__ == "__main__":
    main()