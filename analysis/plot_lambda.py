#!/usr/bin/env python3
"""
plot_lambda.py — 画 λ 敏感性 / 留出验证 / Spearman 三张图。

输入：
  results/lambda_sensitivity.csv
  results/p2rank_pocket_data.csv

输出：
  results/figs/fig_lambda_sensitivity.png
  results/figs/fig_holdout.png
  results/figs/fig_spearman.png
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


# ============================================================
# 工具
# ============================================================
def rank_of(values, descending=False):
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i], reverse=descending)
    ranks = [0] * n
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def select_pocket(pockets, lam):
    valid = [p for p in pockets if p["vina_energy"] != ""]
    if not valid:
        return None
    rank_e = rank_of([p["vina_energy"] for p in valid], descending=False)
    rank_s = rank_of([p["p2rank_score"] for p in valid], descending=True)
    combined = [rank_e[i] + lam * rank_s[i] for i in range(len(valid))]
    return valid[int(np.argmin(combined))]["err_to_true"]


def eval_lambda(by_case, lam, cases=None):
    if cases is None:
        cases = list(by_case.keys())
    errs = []
    for c in cases:
        e = select_pocket(by_case[c], lam)
        if e is not None:
            errs.append(e)
    return np.array(errs) if errs else np.array([])


def load_pocket_data(path):
    by_case = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                by_case[r["case"]].append({
                    "pocket": r["pocket"],
                    "p2rank_score": float(r["p2rank_score"]),
                    "vina_energy": float(r["vina_energy"]) if r["vina_energy"] else "",
                    "err_to_true": float(r["err_to_true"]),
                })
            except (ValueError, KeyError):
                continue
    return by_case


# ============================================================
# Fig 1：λ 敏感性
# ============================================================
def fig_sensitivity(lambdas, rates):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(lambdas, rates, "o-", color="#1f77b4",
            linewidth=2, markersize=8, markeredgecolor="k",
            markeredgewidth=0.5)

    # 标最优
    best_i = int(np.argmax(rates))
    ax.plot(lambdas[best_i], rates[best_i], "o",
            color="#d62728", markersize=14, zorder=5)
    ax.annotate(f"λ* = {lambdas[best_i]:.2f}\n({rates[best_i]:.2f}%)",
                xy=(lambdas[best_i], rates[best_i]),
                xytext=(lambdas[best_i] + 0.4, rates[best_i] - 2),
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.0))

    ax.set_xlabel("λ (weight of P2Rank score rank)", fontsize=13)
    ax.set_ylabel("Success rate (<4 Å, %)", fontsize=13)
    ax.set_title("RankFusion λ sensitivity on P2Rank",
                 fontsize=13, pad=15)
    ax.grid(True, alpha=0.3, lw=0.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)

    fig.tight_layout()
    p = OUT / "fig_lambda_sensitivity.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {p}")


# ============================================================
# Fig 2：留出验证
# ============================================================
def fig_holdout(by_case, lambdas, seed=42):
    cases = sorted(by_case.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    train, test = cases[:280], cases[280:]

    train_rates, test_rates = [], []
    for lam in lambdas:
        t = eval_lambda(by_case, lam, train)
        v = eval_lambda(by_case, lam, test)
        train_rates.append((t < 4).mean() * 100 if len(t) else 0)
        test_rates.append((v < 4).mean() * 100 if len(v) else 0)

    best_lam = lambdas[int(np.argmax(train_rates))]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(lambdas, train_rates, "o-", color="#2ca02c",
            label="Train (n=280)", linewidth=2, markersize=7)
    ax.plot(lambdas, test_rates, "s-", color="#d62728",
            label="Test (n=143)", linewidth=2, markersize=7)

    ax.axvline(best_lam, color="k", ls="--", lw=1.0, alpha=0.6)
    ax.annotate(f"Train-optimal\nλ = {best_lam:.2f}",
                xy=(best_lam, max(train_rates)),
                xytext=(best_lam + 0.4, max(train_rates) - 3),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.0))

    ax.set_xlabel("λ", fontsize=13)
    ax.set_ylabel("Success rate (<4 Å, %)", fontsize=13)
    ax.set_title("Hold-out validation of λ", fontsize=13, pad=15)
    ax.legend(fontsize=11, frameon=False, loc="best")
    ax.grid(True, alpha=0.3, lw=0.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)

    fig.tight_layout()
    p = OUT / "fig_holdout.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {p}")


# ============================================================
# Fig 3：Spearman 散点
# ============================================================
def fig_spearman(by_case):
    s, e = [], []
    for pockets in by_case.values():
        for p in pockets:
            if p["vina_energy"] != "":
                s.append(p["p2rank_score"])
                e.append(p["vina_energy"])

    if len(s) < 5:
        print("  [WARN] 数据不足")
        return

    s = np.array(s); e = np.array(e)
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(s, e)
    except ImportError:
        rho, pval = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(s, e, s=4, alpha=0.25, c="#1f77b4", edgecolors="none")

    # 拟合线（可选）
    try:
        z = np.polyfit(s, e, 1)
        xs = np.linspace(s.min(), s.max(), 100)
        ax.plot(xs, np.polyval(z, xs), "r--", lw=1.2, alpha=0.7)
    except Exception:
        pass

    ax.set_xlabel("P2Rank score", fontsize=13)
    ax.set_ylabel("Vina energy (kcal/mol)", fontsize=13)
    ax.set_title(f"P2Rank score vs Vina energy\n"
                 f"Spearman ρ = {rho:+.3f}, p = {pval:.1e}  (n = {len(s)})",
                 fontsize=12, pad=15)
    ax.grid(True, alpha=0.25, lw=0.4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold"); lbl.set_fontsize(11)

    fig.tight_layout()
    p = OUT / "fig_spearman.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {p}")


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket_csv", default="results/p2rank_pocket_data.csv")
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lambda_list = [float(x) for x in args.lambdas.split(",")]

    print("加载数据 ...")
    by_case = load_pocket_data(args.pocket_csv)
    print(f"  case 数: {len(by_case)}")

    # 全量 λ 扫描
    print("\n全量 λ 扫描:")
    rates = []
    for lam in lambda_list:
        arr = eval_lambda(by_case, lam)
        r = (arr < 4).mean() * 100 if len(arr) else 0
        rates.append(r)
        print(f"  λ = {lam:<5.2f}  <4Å = {r:.2f}%  (n={len(arr)})")

    print("\n画图:")
    fig_sensitivity(lambda_list, rates)
    fig_holdout(by_case, lambda_list, seed=args.seed)
    fig_spearman(by_case)

    print(f"\n完成，图在 {OUT.resolve()}")


if __name__ == "__main__":
    main()