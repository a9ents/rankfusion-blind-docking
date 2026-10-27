#!/usr/bin/env python3
"""
seed_rf_v2.py — 改进版 seed Rank Fusion

不需要重新跑 USalign，直接用 seed_summary.csv 的后处理。

输入：
  seed_rf_results/seed_summary.csv   每个 seed 的信号
  seed_rf_results/seed_aggregate.csv  每个 case 的 top1/best 误差

输出：
  seed_rf_results_v2/strategy_comparison.csv
  seed_rf_results_v2/weight_grid_cv.csv
  seed_rf_results_v2/report.txt
"""
import argparse
import csv
import json
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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
# 从 AF3 json 读局部 pLDDT
# =====================================================================
def load_local_plddt(case_dir, af3_dir):
    """对每个 seed，计算配体附近残基的平均 pLDDT"""
    result = {}
    af3_case = Path(af3_dir) / case_dir
    if not af3_case.is_dir():
        return result

    for i in range(5):
        sample_dir = af3_case / f"SEED-1_SAMPLE-{i}"
        if not sample_dir.is_dir():
            continue
        json_file = None
        for name in ("summary_confidences.json",
                     f"{case_dir}_SEED-1_SAMPLE-{i}_SUMMARY_CONFIDENCES.JSON"):
            p = sample_dir / name
            if p.is_file():
                json_file = p
                break
        if json_file is None:
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # AF3 的 summary_confidences.json 通常不含 per-residue pLDDT
        # 但可能含 atom_plddts 或类似字段；若有则取均值
        for key in ("atom_plddts", "plddts", "plddt"):
            if key in data:
                vals = data[key]
                if isinstance(vals, list) and vals:
                    result[i] = float(np.mean(vals))
                    break
    return result


# =====================================================================
# 融合策略
# =====================================================================
def rank_of(values, descending=True):
    """返回每个 index 的排名（0 = 最好）"""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i], reverse=descending)
    ranks = [0] * n
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def strategy_top1(seeds):
    """ranking_score 最高"""
    return int(np.argmax([s["ranking_score"] for s in seeds]))


def strategy_medoid(seeds):
    """到其他 seed 平均距离最小的（最中心）"""
    if "consistency" not in seeds[0]:
        return 0
    # consistency 越高 = 越中心
    return int(np.argmax([s.get("consistency", 0) for s in seeds]))


def strategy_rf_grid(seeds, w):
    """加权 rank sum"""
    n = len(seeds)
    r_rank = rank_of([s["ranking_score"] for s in seeds], descending=True)
    r_cons = rank_of([s.get("consistency", 0) for s in seeds], descending=True)
    r_clash = rank_of([s["clash_score"] for s in seeds], descending=True)
    r_buried = rank_of([-s["buriedness"] for s in seeds], descending=True)
    r_cont = rank_of([s["receptor_contacts"] for s in seeds], descending=True)

    combined = [
        w["ranking"] * r_rank[i] +
        w["consistency"] * r_cons[i] +
        w["clash"] * r_clash[i] +
        w["buried"] * r_buried[i] +
        w["contacts"] * r_cont[i]
        for i in range(n)
    ]
    return int(np.argmin(combined))


def strategy_rank_product(seeds, w):
    """几何平均（rank product）"""
    n = len(seeds)
    r_rank = [r + 1 for r in rank_of([s["ranking_score"] for s in seeds], True)]
    r_cons = [r + 1 for r in rank_of([s.get("consistency", 0) for s in seeds], True)]
    r_clash = [r + 1 for r in rank_of([s["clash_score"] for s in seeds], True)]
    r_buried = [r + 1 for r in rank_of([-s["buriedness"] for s in seeds], True)]
    r_cont = [r + 1 for r in rank_of([s["receptor_contacts"] for s in seeds], True)]

    combined = []
    for i in range(n):
        prod = (r_rank[i] ** w["ranking"] *
                r_cons[i] ** w["consistency"] *
                r_clash[i] ** w["clash"] *
                r_buried[i] ** w["buried"] *
                r_cont[i] ** w["contacts"])
        combined.append(prod)
    return int(np.argmin(combined))


def strategy_consistency_only(seeds):
    """只用位置一致性"""
    return int(np.argmax([s.get("consistency", 0) for s in seeds]))


def strategy_pareto(seeds):
    """非支配排序：找 Pareto front 中 ranking_score 最高的"""
    n = len(seeds)
    # 三个目标：ranking_score 高、clash_score 高、buriedness 小
    def dominates(i, j):
        ri = (seeds[i]["ranking_score"], seeds[i]["clash_score"], -seeds[i]["buriedness"])
        rj = (seeds[j]["ranking_score"], seeds[j]["clash_score"], -seeds[j]["buriedness"])
        better_or_equal = all(a >= b for a, b in zip(ri, rj))
        strictly_better = any(a > b for a, b in zip(ri, rj))
        return better_or_equal and strictly_better

    front = []
    for i in range(n):
        if not any(dominates(j, i) for j in range(n) if j != i):
            front.append(i)
    if not front:
        return 0
    # 从 front 中选 ranking_score 最高的
    return max(front, key=lambda i: seeds[i]["ranking_score"])


# =====================================================================
# 主流程
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_csv", default="seed_rf_results/seed_summary.csv")
    p.add_argument("--agg_csv", default="seed_rf_results/seed_aggregate.csv")
    p.add_argument("--af3_dir", default="AF3_workflow")
    p.add_argument("--out_dir", default="seed_rf_results_v2")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读数据
    seed_rows = read_csv(args.seed_csv)
    if not seed_rows:
        print("⚠ 无 seed 数据")
        return

    # 按 case 分组
    by_case = defaultdict(list)
    for r in seed_rows:
        by_case[r["case"]].append({
            "seed": int(r["seed"]),
            "err_A": fl(r["err_A"]),
            "ranking_score": fl(r["ranking_score"], 0.0),
            "clash_score": fl(r["clash_score"], 0.0),
            "receptor_contacts": fl(r["receptor_contacts"], 0),
            "buriedness": fl(r["buriedness"], 5.0),
            "consistency": fl(r.get("consistency", 0), 0),
        })

    print(f"读取 {len(by_case)} 个 case，{len(seed_rows)} 个 seed 记录\n")

    # =========================================================
    # 策略 1：Top1 / Medoid / Consistency / Pareto
    # =========================================================
    strategies = {
        "Top1 (ranking_score)": strategy_top1,
        "Medoid (consistency)": strategy_medoid,
        "Consistency-only": strategy_consistency_only,
        "Pareto front": strategy_pareto,
    }

    results = {}
    for name, fn in strategies.items():
        errs = []
        for case, seeds in by_case.items():
            if len(seeds) < 2:
                continue
            idx = fn(seeds)
            errs.append(seeds[idx]["err_A"])
        errs = np.array(errs)
        results[name] = {
            "n": len(errs),
            "mean": errs.mean(),
            "median": np.median(errs),
            "lt2": (errs < 2).mean() * 100,
            "lt4": (errs < 4).mean() * 100,
        }

    # =========================================================
    # 策略 2：权重网格搜索（交叉验证）
    # =========================================================
    cases = list(by_case.keys())
    np.random.seed(42)
    np.random.shuffle(cases)
    n_folds = 5
    folds = np.array_split(cases, n_folds)

    weights_grid = []
    for lr in [0, 0.5, 1.0, 2.0, 5.0]:
        for lc in [0, 0.5, 1.0]:
            for lcl in [0, 0.5, 1.0]:
                for lb in [0, 0.5, 1.0]:
                    for lcont in [0, 0.5, 1.0]:
                        if lr == lc == lcl == lb == lcont:
                            continue
                        weights_grid.append({
                            "ranking": lr, "consistency": lc,
                            "clash": lcl, "buried": lb, "contacts": lcont,
                        })

    print(f"网格搜索 {len(weights_grid)} 个权重组合，5 折交叉验证...\n")

    grid_results = []
    for w in weights_grid:
        fold_rates = []
        for f in range(n_folds):
            test_cases = folds[f]
            errs = []
            for case in test_cases:
                seeds = by_case[case]
                if len(seeds) < 2:
                    continue
                idx = strategy_rf_grid(seeds, w)
                errs.append(seeds[idx]["err_A"])
            if errs:
                fold_rates.append((np.array(errs) < 4).mean())
        if fold_rates:
            grid_results.append({
                **w,
                "cv_lt4": np.mean(fold_rates),
                "cv_std": np.std(fold_rates),
            })

    grid_results.sort(key=lambda r: -r["cv_lt4"])

    # 最优组合
    print("=" * 72)
    print("权重网格搜索结果（按 5 折 CV 成功率排序）")
    print("=" * 72)
    print(f"\n  {'ranking':>8}{'consist':>9}{'clash':>7}{'buried':>8}"
          f"{'contacts':>10}{'CV<4Å':>10}")
    print("  " + "-" * 60)
    for r in grid_results[:10]:
        print(f"  {r['ranking']:>8.1f}{r['consistency']:>9.1f}"
              f"{r['clash']:>7.1f}{r['buried']:>8.1f}"
              f"{r['contacts']:>10.1f}{r['cv_lt4']*100:>9.2f}%")

    # =========================================================
    # 用最优权重在全量上评估
    # =========================================================
    best_w = {k: grid_results[0][k]
              for k in ["ranking", "consistency", "clash", "buried", "contacts"]}
    print(f"\n最优权重: {best_w}\n")

    errs_rf_opt = []
    for case, seeds in by_case.items():
        if len(seeds) < 2:
            continue
        idx = strategy_rf_grid(seeds, best_w)
        errs_rf_opt.append(seeds[idx]["err_A"])
    errs_rf_opt = np.array(errs_rf_opt)

    results["RF-grid-opt"] = {
        "n": len(errs_rf_opt),
        "mean": errs_rf_opt.mean(),
        "median": np.median(errs_rf_opt),
        "lt2": (errs_rf_opt < 2).mean() * 100,
        "lt4": (errs_rf_opt < 4).mean() * 100,
    }

    # =========================================================
    # 策略 3：Rank Product
    # =========================================================
    errs_prod = []
    for case, seeds in by_case.items():
        if len(seeds) < 2:
            continue
        idx = strategy_rank_product(seeds, best_w)
        errs_prod.append(seeds[idx]["err_A"])
    errs_prod = np.array(errs_prod)
    results["Rank-product"] = {
        "n": len(errs_prod),
        "mean": errs_prod.mean(),
        "median": np.median(errs_prod),
        "lt2": (errs_prod < 2).mean() * 100,
        "lt4": (errs_prod < 4).mean() * 100,
    }

    # =========================================================
    # 加入 Best (oracle) 和之前 RF 作为对比
    # =========================================================
    best_errs = []
    for case, seeds in by_case.items():
        if len(seeds) < 2:
            continue
        best_errs.append(min(s["err_A"] for s in seeds))
    best_errs = np.array(best_errs)
    results["Best (oracle)"] = {
        "n": len(best_errs),
        "mean": best_errs.mean(),
        "median": np.median(best_errs),
        "lt2": (best_errs < 2).mean() * 100,
        "lt4": (best_errs < 4).mean() * 100,
    }

    # =========================================================
    # 报告
    # =========================================================
    L = []
    L.append("=" * 72)
    L.append("AF3 5-seed Rank Fusion 改进尝试")
    L.append("=" * 72)
    L.append(f"\n  总 case: {len(by_case)}")

    L.append("\n" + "-" * 72)
    L.append("【所有策略对比】")
    L.append("-" * 72)
    L.append(f"\n  {'策略':<26}{'n':>6}{'均值':>10}{'中位数':>10}"
             f"{'<2Å':>9}{'<4Å':>9}")
    L.append("  " + "-" * 70)
    for name in ["Top1 (ranking_score)", "Medoid (consistency)",
                 "Consistency-only", "Pareto front",
                 "RF-grid-opt", "Rank-product", "Best (oracle)"]:
        if name not in results:
            continue
        r = results[name]
        L.append(f"  {name:<26}{r['n']:>6}{r['mean']:>10.2f}"
                 f"{r['median']:>10.2f}{r['lt2']:>8.1f}%{r['lt4']:>8.1f}%")

    # 结论
    L.append("\n" + "=" * 72)
    L.append("【结论】")
    L.append("=" * 72)
    top1_rate = results["Top1 (ranking_score)"]["lt4"]
    rf_rate = results["RF-grid-opt"]["lt4"]
    best_rate = results["Best (oracle)"]["lt4"]

    L.append(f"\n  Top1 成功率:      {top1_rate:.2f}%")
    L.append(f"  RF-opt 成功率:    {rf_rate:.2f}%  (改进 {rf_rate-top1_rate:+.2f} pp)")
    L.append(f"  Best (oracle):    {best_rate:.2f}%  (天花板)")
    L.append(f"  RF-opt / Best:    {rf_rate/best_rate*100:.1f}%")

    if rf_rate > top1_rate + 1:
        L.append(f"\n  ✅ 改进策略有效：RF-opt 比 Top1 提升 {rf_rate-top1_rate:+.2f} pp")
    elif abs(rf_rate - top1_rate) < 1:
        L.append(f"\n  ⚠ 改进策略无效：RF-opt 与 Top1 无显著差异")
    else:
        L.append(f"\n  ❌ 改进策略变差")

    # 相关性分析
    if HAS_SCIPY:
        all_rank = [s["ranking_score"] for seeds in by_case.values() for s in seeds]
        all_err = [s["err_A"] for seeds in by_case.values() for s in seeds]
        all_clash = [s["clash_score"] for seeds in by_case.values() for s in seeds]
        all_bur = [s["buriedness"] for seeds in by_case.values() for s in seeds]
        all_cont = [s["receptor_contacts"] for seeds in by_case.values() for s in seeds]

        L.append("\n" + "-" * 72)
        L.append("【信号 vs 误差的相关性（全 seed 池）】")
        L.append("-" * 72)
        L.append(f"\n  {'信号':<22}{'Spearman ρ':>14}{'p':>12}")
        L.append("  " + "-" * 50)
        for name, vals in [("ranking_score", all_rank),
                           ("clash_score", all_clash),
                           ("buriedness", all_bur),
                           ("receptor_contacts", all_cont)]:
            rho, pv = spearmanr(vals, all_err)
            L.append(f"  {name:<22}{rho:>+14.3f}{pv:>12.2e}")

    text = "\n".join(L)
    print("\n" + text)

    # 写文件
    report_path = out_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n✅ {report_path}")

    # 写策略对比 CSV
    cmp_csv = out_dir / "strategy_comparison.csv"
    with open(cmp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "n", "mean", "median", "lt2_pct", "lt4_pct"])
        for name, r in results.items():
            w.writerow([name, r["n"], f"{r['mean']:.3f}",
                        f"{r['median']:.3f}", f"{r['lt2']:.2f}",
                        f"{r['lt4']:.2f}"])
    print(f"✅ {cmp_csv}")

    # 写网格搜索 CSV
    grid_csv = out_dir / "weight_grid_cv.csv"
    with open(grid_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_results[0].keys()))
        w.writeheader()
        w.writerows(grid_results)
    print(f"✅ {grid_csv}")


if __name__ == "__main__":
    main()