#!/usr/bin/env python3
"""
plot_probe_v7.py — fig1/fig2/fig3 图例移到标题下方
"""
import csv
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["font.weight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "bold"
matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path("probe_results/figs")
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300


def read_csv(path):
    p = Path(path)
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fl(v, default=None):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# =====================================================================
# fig1：探针半径分布直方图
# =====================================================================
def fig_probe_hist(rows):
    pa = np.array([fl(r["probe_af3_A"]) for r in rows
                   if fl(r["probe_af3_A"]) is not None])
    pe = np.array([fl(r["probe_exp_A"]) for r in rows
                   if fl(r["probe_exp_A"]) is not None])

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    bins = np.arange(0, 5.5, 0.25)

    h1 = ax.hist(pe, bins=bins, color="#2ca02c", alpha=0.55,
                 edgecolor="k", linewidth=0.4)
    h2 = ax.hist(pa, bins=bins, color="#d62728", alpha=0.65,
                 edgecolor="k", linewidth=0.4)

    v1 = ax.axvline(1.4, color="gray", ls="--", lw=1.2)
    v2 = ax.axvline(1.8, color="blue", ls=":", lw=1.4)

    ax.set_xlabel("Max probe radius (\u00c5)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count", fontsize=13, fontweight="bold")
    ax.set_title("Probe radius distribution at true pocket center",
                 fontsize=13, fontweight="bold", pad=55)

    # ★ 图例统一放到标题下方
    handles = [h1[2][0], h2[2][0], v1, v2]
    labels = ["Experimental", "AF3",
              "Water = 1.4 \u00c5", "fpocket probe = 1.8 \u00c5"]
    leg = ax.legend(handles, labels,
                    loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=4, frameon=False, fontsize=11,
                    handlelength=2.0, columnspacing=2.5,
                    handletextpad=0.7)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_probe_hist.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {OUT / 'fig1_probe_hist.png'}")


# =====================================================================
# fig2：AF3 vs 实验 配对散点
# =====================================================================
def fig_probe_paired(rows):
    xs, ys = [], []
    for r in rows:
        a = fl(r["probe_af3_A"]); e = fl(r["probe_exp_A"])
        if a is None or e is None:
            continue
        xs.append(e); ys.append(a)

    if not xs:
        print("  ⚠ fig2: 无数据"); return

    xs = np.array(xs); ys = np.array(ys)

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    s1 = ax.scatter(xs, ys, s=14, c="#9467bd", alpha=0.5,
                    edgecolors="none")
    lim = max(xs.max(), ys.max()) * 1.05
    v1, = ax.plot([0, lim], [0, lim], "k--", lw=0.9)
    v2 = ax.axvline(1.8, color="blue", ls=":", lw=1.2)
    v3 = ax.axhline(1.8, color="blue", ls=":", lw=1.2)

    # Spearman 与图例并列
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(xs, ys)
        spearman_label = f"Spearman \u03c1 = {rho:+.3f},  p = {p:.2e}"
    except ImportError:
        spearman_label = ""

    from matplotlib.patches import Patch
    handles = [s1, v1, v2,
               Patch(facecolor="none", edgecolor="none")]
    labels = ["Cases", "y = x",
              "fpocket threshold = 1.8 \u00c5",
              spearman_label]
    leg = ax.legend(handles, labels,
                    loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=4, frameon=False, fontsize=10.5,
                    handlelength=2.0, columnspacing=2.5,
                    handletextpad=0.7)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    ax.set_xlabel("Experimental probe radius (\u00c5)",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel("AF3 probe radius (\u00c5)",
                  fontsize=13, fontweight="bold")
    ax.set_title("AF3 vs experimental probe radius at true pocket",
                 fontsize=13, fontweight="bold", pad=55)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)
    ax.grid(True, alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_probe_paired.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {OUT / 'fig2_probe_paired.png'}")


# =====================================================================
# fig3：探针半径分档柱状图
# =====================================================================
def fig_probe_bins(rows):
    pa = np.array([fl(r["probe_af3_A"]) for r in rows
                   if fl(r["probe_af3_A"]) is not None])
    pe = np.array([fl(r["probe_exp_A"]) for r in rows
                   if fl(r["probe_exp_A"]) is not None])

    bins = [(0, 1), (1, 2), (2, 3), (3, 5)]
    labels = ["[0, 1)\nblocked", "[1, 2)\nwater-tight",
              "[2, 3)\nmarginal", "[3, 5)\nfitting"]
    af3_c = [int(((pa >= lo) & (pa < hi)).sum()) for lo, hi in bins]
    exp_c = [int(((pe >= lo) & (pe < hi)).sum()) for lo, hi in bins]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    b1 = ax.bar(x - w/2, af3_c, w, color="#d62728",
                alpha=0.85, edgecolor="k", linewidth=0.6)
    b2 = ax.bar(x + w/2, exp_c, w, color="#2ca02c",
                alpha=0.85, edgecolor="k", linewidth=0.6)

    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                f"{int(b.get_height())}", ha="center",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Probe radius (\u00c5)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of cases", fontsize=13, fontweight="bold")
    ax.set_title("Probe radius bins: AF3 vs experimental",
                 fontsize=13, fontweight="bold", pad=50)

    # ★ 图例统一放到标题下方
    leg = ax.legend([b1[0], b2[0]], ["AF3", "Experimental"],
                    loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=2, frameon=False, fontsize=11,
                    handlelength=2.0, columnspacing=3.0)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_probe_bins.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {OUT / 'fig3_probe_bins.png'}")


def main():
    print("重画 fig1 / fig2 / fig3（图例移到标题下方）...\n")
    rows = read_csv("probe_results/probe_summary.csv")
    rows = [r for r in rows if r.get("status") == "ok"]
    if not rows:
        print("⚠ 无数据"); return
    print(f"有效 case: {len(rows)}\n")
    fig_probe_hist(rows)
    fig_probe_paired(rows)
    fig_probe_bins(rows)
    print(f"\n完成。")


if __name__ == "__main__":
    main()