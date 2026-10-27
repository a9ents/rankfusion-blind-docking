#!/usr/bin/env python3
"""
fig2_lambda.py — 重画图 2：fpocket 和 P2Rank 的 λ 敏感性曲线。

输入：
  results/fpocket_lambda_sensitivity.csv
  results/lambda_sensitivity.csv

输出：
  results/figs/fig2_lambda_sensitivity.png
"""
import csv
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
DPI = 600


def read_lambda_csv(path, rate_col):
    """读 lambda CSV，返回 (lambdas, rates_%)。"""
    lam, rate = [], []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                lam.append(float(r["lambda"]))
                rate.append(float(r[rate_col]) * 100)
            except (KeyError, ValueError):
                continue
    return np.array(lam), np.array(rate)


def main():
    # --- 读数据 ---
    lam_fp, rate_fp = read_lambda_csv(
        "results/fpocket_lambda_sensitivity.csv", "lt_4A")
    lam_p2, rate_p2 = read_lambda_csv(
        "results/lambda_sensitivity.csv", "lt_4A")

    print("fpocket:")
    for l, r in zip(lam_fp, rate_fp):
        print(f"  λ={l:<5.2f}  {r:.2f}%")
    print("\nP2Rank:")
    for l, r in zip(lam_p2, rate_p2):
        print(f"  λ={l:<5.2f}  {r:.2f}%")

    # --- 画图 ---
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(lam_fp, rate_fp, "o-", color="#2ca02c",
            linewidth=2, markersize=8, markeredgecolor="k",
            markeredgewidth=0.6, label="fpocket")
    ax.plot(lam_p2, rate_p2, "s-", color="#d62728",
            linewidth=2, markersize=8, markeredgecolor="k",
            markeredgewidth=0.6, label="P2Rank")

    # 标记 λ=0.75
    ax.axvline(0.75, color="k", ls="--", lw=1.0, alpha=0.5)
    ax.text(0.78, ax.get_ylim()[0] + 0.5, "λ = 0.75",
            fontsize=10, fontweight="bold", va="bottom")

    # 标记 λ=0
    ax.axvline(0.0, color="gray", ls=":", lw=0.8, alpha=0.5)

    ax.set_xlabel("λ (weight of detector-score rank)", fontsize=13)
    ax.set_ylabel("Success rate (<4 Å, %)", fontsize=13)
    ax.set_title("RankFusion λ sensitivity on fpocket and P2Rank",
                 fontsize=13, pad=35)

    # 图注放到标题下方的中间
    ax.legend(fontsize=12, frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))

    ax.grid(True, alpha=0.3, lw=0.5)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)

    fig.tight_layout()
    out = OUT / "fig2_lambda_sensitivity.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()