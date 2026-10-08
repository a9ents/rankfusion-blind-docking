#!/usr/bin/env python3
"""
fig_holdout_lambda.py — Hold-out validation of λ（图注移到标题下方居中）。
"""
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


def main():
    lam = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    train = np.array([45.4, 45.8, 46.2, 46.2, 41.1, 40.4, 40.4, 39.7])
    test  = np.array([40.7, 40.0, 40.0, 40.0, 40.7, 40.7, 40.7, 39.3])

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(lam, train, "o-", color="#2ca02c",
            linewidth=2.5, markersize=11, markeredgecolor="#2ca02c",
            label="Train (n=280)")
    ax.plot(lam, test, "s-", color="#d62728",
            linewidth=2.5, markersize=11, markeredgecolor="#d62728",
            label="Test (n=143)")

    # λ = 0.50 竖线 + 注释
    ax.axvline(0.5, color="k", ls="--", lw=1.0, alpha=0.6)
    ax.annotate("Train-optimal\nλ = 0.50",
                xy=(0.5, 46.2), xytext=(0.95, 43.4),
                fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.2, color="k"))

    ax.set_xlabel("λ", fontsize=15)
    ax.set_ylabel("Success rate (<4 Å, %)", fontsize=15)
    ax.set_title("Hold-out validation of λ", fontsize=17, pad=45)

    # 图注放到标题下方的中间
    ax.legend(fontsize=14, frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))

    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(39, 46.5)
    ax.grid(True, alpha=0.3, lw=0.6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(12)

    fig.tight_layout()
    p = OUT / "fig_holdout_lambda.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p}")


if __name__ == "__main__":
    main()