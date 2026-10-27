#!/usr/bin/env python3
"""stats_analysis.py — 论文统计检验（Wilcoxon / McNemar / Bootstrap）。"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load_eval(eval_csv):
    # 强制 UTF-8，兼容 BOM
    with open(eval_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # 去掉列名前导/后置空格
    cleaned = []
    for r in rows:
        cleaned.append({(k.strip() if k else k): v for k, v in r.items()})
    return cleaned


def extract(rows, method):
    """返回 (cases, errors)；跳过空值。"""
    cases, errs = [], []
    key = f"{method}_err"
    for r in rows:
        v = r.get(key, "")
        if v in ("", None):
            continue
        try:
            errs.append(float(v))
            cases.append(r["case"])
        except (ValueError, TypeError):
            continue
    return cases, errs


def wilcoxon_test(a, b, name_a, name_b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) != len(b) or len(a) < 5:
        print(f"  [skip wilcoxon {name_a} vs {name_b}] len(a)={len(a)}, len(b)={len(b)}")
        return None
    diff = a - b
    n_eff = int((diff != 0).sum())
    if n_eff < 5:
        print(f"  [skip wilcoxon {name_a} vs {name_b}] n_eff={n_eff}")
        return None
    try:
        stat, p = stats.wilcoxon(a, b, zero_method="wilcox",
                                 alternative="two-sided")
    except Exception as e:
        print(f"  [! wilcoxon {name_a} vs {name_b}] {e}")
        return None
    N_total = len(a)
    try:
        z = stats.norm.isf(p / 2) if p > 0 else 0.0
    except Exception:
        z = 0.0
    r = abs(z) / np.sqrt(N_total) if N_total > 0 else 0.0
    return {
        "comparison": f"{name_a} vs {name_b}",
        "test": "Wilcoxon",
        "N_eff": n_eff,
        "N_total": N_total,
        "stat": float(stat),
        "p": float(p),
        "effect_r": float(r),
    }


def mcnemar_test(a, b, thr, name_a, name_b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) != len(b) or len(a) < 5:
        return None
    hit_a = a < thr
    hit_b = b < thr
    n11 = int(np.sum(hit_a & hit_b))
    n10 = int(np.sum(hit_a & ~hit_b))
    n01 = int(np.sum(~hit_a & hit_b))
    n00 = int(np.sum(~hit_a & ~hit_b))
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        table = [[n11, n10], [n01, n00]]
        res = mcnemar(table, exact=False, correction=True)
        stat, p = float(res.statistic), float(res.pvalue)
    except ImportError:
        if n10 + n01 == 0:
            stat, p = 0.0, 1.0
        else:
            stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
            p = 1 - stats.chi2.cdf(stat, df=1)
    except Exception as e:
        print(f"  [! mcnemar {name_a} vs {name_b}] {e}")
        return None
    return {
        "comparison": f"{name_a} vs {name_b}",
        "test": f"McNemar(<{thr}Å)",
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "stat": float(stat),
        "p": float(p),
    }


def bootstrap_ci(a, b, n_boot=10000, seed=42):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) != len(b) or len(a) < 5:
        return None
    rng = np.random.default_rng(seed)
    diffs = a - b
    n = len(diffs)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = np.mean(diffs[idx])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_csv", default="results/evaluation.csv")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    rows = load_eval(args.eval_csv)
    print(f"读取 {len(rows)} 行")
    if rows:
        print(f"列名: {list(rows[0].keys())}")

    # 检查各方法数据量
    print("\n各方法非空样本数:")
    for m in ["exp_top1", "exp_energy", "exp_rankfusion",
              "p2rank_top1", "p2rank_energy", "p2rank_rf",
              "af3top1_rankfusion", "af3oracle_rankfusion", "af3ligand"]:
        c, e = extract(rows, m)
        print(f"  {m:25s}: {len(e)}")

    comparisons = [
        ("af3top1_rankfusion", "exp_top1"),
        ("af3top1_rankfusion", "exp_energy"),
        ("af3top1_rankfusion", "exp_rankfusion"),
        ("af3top1_rankfusion", "af3ligand"),
        ("af3top1_rankfusion", "af3oracle_rankfusion"),
        ("exp_rankfusion", "exp_top1"),
        ("exp_rankfusion", "exp_energy"),
        ("exp_rankfusion", "p2rank_top1"),
        ("exp_top1", "p2rank_top1"),
        ("exp_rankfusion", "af3ligand"),
        ("p2rank_top1", "p2rank_rf"),
        ("p2rank_rf", "p2rank_top1"),
        ("p2rank_top1", "p2rank_energy"),
        ("p2rank_energy", "p2rank_rf"),
        ("exp_rankfusion", "p2rank_rf"),
        ("exp_rankfusion", "p2rank_energy"),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== 配对检验 =====")
    pairwise_rows = []
    for a, b in comparisons:
        ca, ea = extract(rows, a)
        cb, eb = extract(rows, b)
        common = sorted(set(ca) & set(cb))
        ea_map = dict(zip(ca, ea))
        eb_map = dict(zip(cb, eb))
        ea_al = [ea_map[c] for c in common]
        eb_al = [eb_map[c] for c in common]

        print(f"\n{a} vs {b}: common={len(common)}")

        w = wilcoxon_test(ea_al, eb_al, a, b)
        if w:
            print(f"  Wilcoxon: N={w['N_total']} N_eff={w['N_eff']} "
                  f"p={w['p']:.6g} r={w['effect_r']:.3f}")
            pairwise_rows.append(w)

        m = mcnemar_test(ea_al, eb_al, 4.0, a, b)
        if m:
            print(f"  McNemar:  n10={m['n10']} n01={m['n01']} p={m['p']:.6g}")
            pairwise_rows.append(m)

    if pairwise_rows:
        keys = ["comparison", "test", "N_eff", "N_total",
                "n11", "n10", "n01", "n00", "stat", "p", "effect_r"]
        with open(out_dir / "stats_pairwise.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in pairwise_rows:
                w.writerow(r)
        print(f"\n✅ {out_dir / 'stats_pairwise.csv'}")

    # Bootstrap
    boot_rows = []
    for a, b in comparisons:
        ca, ea = extract(rows, a)
        cb, eb = extract(rows, b)
        common = sorted(set(ca) & set(cb))
        ea_map = dict(zip(ca, ea))
        eb_map = dict(zip(cb, eb))
        ea_al = [ea_map[c] for c in common]
        eb_al = [eb_map[c] for c in common]
        r = bootstrap_ci(np.array(ea_al), np.array(eb_al))
        if r:
            boot_rows.append({
                "comparison": f"{a} vs {b}",
                "mean_diff": round(r["mean_diff"], 3),
                "ci_low": round(r["ci_low"], 3),
                "ci_high": round(r["ci_high"], 3),
                "n_boot": r["n_boot"],
                "significant": "yes" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "no",
            })
    if boot_rows:
        with open(out_dir / "stats_bootstrap_ci.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
            w.writeheader()
            w.writerows(boot_rows)
        print(f"✅ {out_dir / 'stats_bootstrap_ci.csv'}")


if __name__ == "__main__":
    main()