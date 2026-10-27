#!/usr/bin/env python3
"""
plot_oracle_v2.py — Oracle 结果画图（图例统一在标题下方）

输出（oracle_results/figs/）：
  fig1_three_methods.png        三种方法成功率对比
  fig2_oracle_vs_global.png     Oracle vs 全局对齐 配对散点
  fig3_oracle_vs_rmsd.png       Oracle 误差 vs AF3 全局 RMSD
  fig4_oracle_dist.png          Oracle 误差分布
  fig5_iptm_bins.png            Oracle 按 ipTM 分档
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["font.weight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "bold"
matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path("oracle_results/figs")
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


def _invisible():
    return Patch(facecolor="none", edgecolor="none")


# =====================================================================
# fig1：三种方法成功率对比（无图例，改标题 pad 保持一致）
# =====================================================================
def fig_three_methods(oracle_ok):
    hits = np.array([fl(r["hit_4A"]) for r in oracle_ok])
    oracle_rate = hits.mean() * 100

    methods = ["Experimental\n(Ref.)",
               "AF3 global\nalignment",
               "AF3 oracle\nlocal alignment"]
    rates = [36.2, 4.97, oracle_rate]
    colors = ["#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.bar(methods, rates, color=colors, alpha=0.85,
                  edgecolor="k", linewidth=0.8)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                f"{r:.2f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Success rate (<4 \u00c5, %)", fontsize=13, fontweight="bold")
    ax.set_title("Rank Fusion success rate on different structures",
                 fontsize=13, fontweight="bold", pad=30)
    ax.set_ylim(0, max(rates) * 1.25)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_three_methods.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {OUT / 'fig1_three_methods.png'}")


# =====================================================================
# fig2：Oracle vs 全局对齐 配对散点
# =====================================================================
def fig_oracle_vs_global(oracle_ok):
    ev = read_csv("results/evaluation.csv")
    global_map = {}
    for r in ev:
        v = r.get("af3top1_rankfusion_err", "")
        if v:
            global_map[r["case"]] = fl(v)

    xs, ys, hit = [], [], []
    for r in oracle_ok:
        case = r["case"]
        if case in global_map:
            xs.append(global_map[case])
            ys.append(fl(r["err_A"]))
            hit.append(int(fl(r["hit_4A"])))
    if not xs:
        print("  ⚠ fig2: 无配对数据"); return
    xs = np.array(xs); ys = np.array(ys); hit = np.array(hit)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    s0 = ax.scatter(xs[hit == 0], ys[hit == 0], s=16, c="#999999",
                    alpha=0.5, edgecolors="none")
    s1 = ax.scatter(xs[hit == 1], ys[hit == 1], s=32, c="#d62728",
                    alpha=0.85, edgecolors="k", linewidths=0.4)

    lim = max(xs.max(), ys.max()) * 1.05
    diag, = ax.plot([0, lim], [0, lim], "k--", lw=0.9)
    ax.axhline(4, color="#1f77b4", ls=":", lw=1.0)
    ax.axvline(4, color="#1f77b4", ls=":", lw=1.0)

    # Spearman
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(xs, ys)
        sp_label = f"Spearman \u03c1 = {rho:+.3f},  p = {p:.2e}"
    except ImportError:
        sp_label = ""

    handles = [s0, s1, diag, _invisible()]
    labels = ["Miss", "Hit (<4 \u00c5)", "y = x", sp_label]

    leg = ax.legend(handles, labels,
                    loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=4, frameon=False, fontsize=10.5,
                    handlelength=2.0, columnspacing=2.5,
                    handletextpad=0.7)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    ax.set_xlabel("Global alignment error (\u00c5)",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel("Oracle local alignment error (\u00c5)",
                  fontsize=13, fontweight="bold")
    ax.set_title("Oracle vs Global alignment",
                 fontsize=13, fontweight="bold", pad=55)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)
    ax.grid(True, alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_oracle_vs_global.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {OUT / 'fig2_oracle_vs_global.png'}")


# =====================================================================
# fig3：Oracle 误差 vs AF3 全局 RMSD
# =====================================================================
def fig_oracle_vs_rmsd(oracle_ok):
    af3 = read_csv("results/af3_model_summary.csv")
    rmsd_map = {}
    for r in af3:
        if r.get("model") == "af3_top1":
            rmsd_map[r["case"]] = fl(r.get("rmsd_to_exp"))

    xs, ys, hit = [], [], []
    for r in oracle_ok:
        case = r["case"]
        if case in rmsd_map:
            xs.append(rmsd_map[case])
            ys.append(fl(r["err_A"]))
            hit.append(int(fl(r["hit_4A"])))
    if not xs:
        print("  ⚠ fig3: 无数据"); return
    xs = np.array(xs); ys = np.array(ys); hit = np.array(hit)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    s0 = ax.scatter(xs[hit == 0], ys[hit == 0], s=16, c="#999999",
                    alpha=0.5, edgecolors="none")
    s1 = ax.scatter(xs[hit == 1], ys[hit == 1], s=32, c="#d62728",
                    alpha=0.85, edgecolors="k", linewidths=0.4)
    h1 = ax.axhline(4, color="#1f77b4", ls=":", lw=1.0)

    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(xs, ys)
        sp_label = f"Spearman \u03c1 = {rho:+.3f},  p = {p:.2e}"
    except ImportError:
        sp_label = ""

    handles = [s0, s1, h1, _invisible()]
    labels = ["Miss", "Hit (<4 \u00c5)", "4 \u00c5", sp_label]

    leg = ax.legend(handles, labels,
                    loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=4, frameon=False, fontsize=10.5,
                    handlelength=2.0, columnspacing=2.5,
                    handletextpad=0.7)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    ax.set_xlabel("AF3 global RMSD (\u00c5)",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel("Oracle local alignment error (\u00c5)",
                  fontsize=13, fontweight="bold")
    ax.set_title("Oracle error vs AF3 structural accuracy",
                 fontsize=13, fontweight="bold", pad=55)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)
    ax.grid(True, alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_oracle_vs_rmsd.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {OUT / 'fig3_oracle_vs_rmsd.png'}")


# =====================================================================
# fig4：Oracle 误差分布
# =====================================================================
def fig_oracle_dist(oracle_ok):
    errs = np.array([fl(r["err_A"]) for r in oracle_ok])
    if len(errs) == 0:
        print("  ⚠ fig4: 无数据"); return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    h = ax.hist(errs, bins=np.arange(0, 80, 4), color="#9467bd",
                alpha=0.8, edgecolor="k", linewidth=0.5)
    v1 = ax.axvline(4, color="#d62728", ls="--", lw=1.2)
    v2 = ax.axvline(np.median(errs), color="#2ca02c", ls=":", lw=1.2)

    ax.set_xlabel("Oracle local alignment error (\u00c5)",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel("Count", fontsize=13, fontweight="bold")
    ax.set_title("Distribution of oracle alignment error",
                 fontsize=13, fontweight="bold", pad=50)

    handles = [h[2][0], v1, v2]
    labels = ["Cases", "4 \u00c5",
              f"Median = {np.median(errs):.2f} \u00c5"]

    leg = ax.legend(handles, labels,
                    loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=3, frameon=False, fontsize=11,
                    handlelength=2.0, columnspacing=3.0,
                    handletextpad=0.7)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_oracle_dist.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {OUT / 'fig4_oracle_dist.png'}")


# =====================================================================
# fig5：按 ipTM 分档
# =====================================================================
def fig_iptm_bins(oracle_ok):
    af3 = read_csv("results/af3_model_summary.csv")
    iptm_map = {}
    for r in af3:
        if r.get("model") == "af3_top1":
            iptm_map[r["case"]] = fl(r.get("iptm"))

    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    labels, rates, ns = [], [], []
    for lo, hi in bins:
        sub = [r for r in oracle_ok
               if r["case"] in iptm_map
               and lo <= iptm_map[r["case"]] <= hi]
        if not sub:
            continue
        h = np.array([fl(r["hit_4A"]) for r in sub])
        labels.append(f"[{lo}, {hi})")
        rates.append(h.mean() * 100)
        ns.append(len(sub))

    if not labels:
        print("  ⚠ fig5: 无数据"); return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(range(len(labels)), rates, color="#9467bd",
                  alpha=0.85, edgecolor="k", linewidth=0.6)
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"n={n}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right",
                       fontsize=11, fontweight="bold")
    ax.set_xlabel("AF3 ipTM", fontsize=13, fontweight="bold")
    ax.set_ylabel("Oracle success rate (<4 \u00c5, %)",
                  fontsize=13, fontweight="bold")
    ax.set_title("Oracle alignment success vs AF3 confidence",
                 fontsize=13, fontweight="bold", pad=30)
    ax.set_ylim(0, max(rates) * 1.25 + 2)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_iptm_bins.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {OUT / 'fig5_iptm_bins.png'}")


# =====================================================================
# 主流程
# =====================================================================
def main():
    print("生成 Oracle 结果图 ...\n")

    oracle = read_csv("oracle_results/oracle_summary.csv")
    ok = [r for r in oracle if r.get("status") == "ok"]
    if not ok:
        print("⚠ 无 oracle 成功数据"); return

    print(f"Oracle 成功 case: {len(ok)}")
    print(f"输出: {OUT.resolve()}\n")

    fig_three_methods(ok)
    fig_oracle_vs_global(ok)
    fig_oracle_vs_rmsd(ok)
    fig_oracle_dist(ok)
    fig_iptm_bins(ok)

    print(f"\n完成。")


if __name__ == "__main__":
    main()