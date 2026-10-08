#!/usr/bin/env python3
"""
supplementary_stats.py — 补充分统计分析

包含四项：
  1. Fisher 精确检验：ipTM 高/低档成功率差异
  2. 成功率置信区间（Wilson / Clopper-Pearson）
  3. 残基配对 vs 未配对（Fisher 检验）
  4. PDB 发布日期自查（训练集泄漏）

输出：
  results/supplementary_stats.txt
  results/pdb_release_dates.csv

用法：
  python supplementary_stats.py
  python supplementary_stats.py --skip-pdb
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, beta

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


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
# 1. Fisher 精确检验
# =====================================================================
def fisher_tests(rows):
    lines = []
    lines.append("=" * 70)
    lines.append("1. Fisher 精确检验：ipTM 分档成功率差异")
    lines.append("=" * 70)

    def count(lo, hi, include_hi=False):
        n_hit = n_miss = 0
        for r in rows:
            iptm = fl(r.get("iptm"))        # ← 小写
            hit = fl(r.get("hit_4A"))
            if iptm is None or hit is None:
                continue
            if include_hi:
                in_bin = lo <= iptm <= hi
            else:
                in_bin = lo <= iptm < hi
            if not in_bin:
                continue
            if int(hit) == 1:
                n_hit += 1
            else:
                n_miss += 1
        return n_hit, n_miss

    hi_hit, hi_miss = count(0.8, 1.0, include_hi=True)
    lo_hit, lo_miss = count(0.0, 0.4, include_hi=False)

    hi_n = hi_hit + hi_miss
    lo_n = lo_hit + lo_miss
    hi_rate = hi_hit / hi_n * 100 if hi_n > 0 else 0.0
    lo_rate = lo_hit / lo_n * 100 if lo_n > 0 else 0.0

    lines.append(f"\n【ipTM ≥ 0.8】  n={hi_n}  hit={hi_hit}  rate={hi_rate:.2f}%")
    lines.append(f"【ipTM < 0.4】  n={lo_n}  hit={lo_hit}  rate={lo_rate:.2f}%")

    if hi_n > 0 and lo_n > 0:
        odds, p = fisher_exact([[hi_hit, hi_miss], [lo_hit, lo_miss]])
        lines.append(f"\n  Fisher exact:  p = {p:.6g}  odds ratio = {odds:.3f}")
        if p < 0.001:
            lines.append("  → 显著 (p < 0.001)")
        elif p < 0.01:
            lines.append("  → 显著 (p < 0.01)")
        elif p < 0.05:
            lines.append("  → 显著 (p < 0.05)")

    # 相邻档对比
    lines.append("\n【各档 vs 最低档 [0, 0.2)】")
    base_hit, base_miss = count(0.0, 0.2, include_hi=False)
    for lo, hi in [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]:
        h, m = count(lo, hi)
        if (h + m) > 0 and (base_hit + base_miss) > 0:
            odds, p = fisher_exact([[h, m], [base_hit, base_miss]])
            lines.append(f"  [{lo}, {hi}) vs [0, 0.2):  "
                         f"hit={h}/{h+m}  p = {p:.4g}  OR = {odds:.2f}")

    return lines


# =====================================================================
# 2. 置信区间
# =====================================================================
def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0
    z = 1.959963985
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def clopper_pearson_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def ci_report(rows):
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("2. 成功率置信区间")
    lines.append("=" * 70)

    def stats_for(lo_iptm, hi_iptm, include_hi=False, label=""):
        hits = []
        for r in rows:
            iptm = fl(r.get("iptm"))
            hit = fl(r.get("hit_4A"))
            if iptm is None or hit is None:
                continue
            if include_hi:
                ok = lo_iptm <= iptm <= hi_iptm
            else:
                ok = lo_iptm <= iptm < hi_iptm
            if ok:
                hits.append(int(hit))
        if not hits:
            return None
        k = sum(hits)
        n = len(hits)
        p = k / n * 100
        w_lo, w_hi = wilson_ci(k, n)
        cp_lo, cp_hi = clopper_pearson_ci(k, n)
        return {
            "label": label, "n": n, "k": k, "rate": p,
            "wilson_lo": w_lo * 100, "wilson_hi": w_hi * 100,
            "cp_lo": cp_lo * 100, "cp_hi": cp_hi * 100,
        }

    bins = [
        (0.0, 0.2, False, "ipTM < 0.2"),
        (0.2, 0.4, False, "0.2 ≤ ipTM < 0.4"),
        (0.4, 0.6, False, "0.4 ≤ ipTM < 0.6"),
        (0.6, 0.8, False, "0.6 ≤ ipTM < 0.8"),
        (0.8, 1.0, True,  "ipTM ≥ 0.8"),
    ]

    lines.append(f"\n{'区间':<20}{'n':>5}{'hit':>5}{'rate':>9}"
                 f"{'Wilson 95%':>20}{'CP 95%':>20}")
    for lo, hi, inc, label in bins:
        s = stats_for(lo, hi, inc, label)
        if s is None:
            continue
        lines.append(
            f"{s['label']:<20}{s['n']:>5}{s['k']:>5}{s['rate']:>8.2f}%"
            f"   [{s['wilson_lo']:>5.1f}%, {s['wilson_hi']:>5.1f}%]"
            f"   [{s['cp_lo']:>5.1f}%, {s['cp_hi']:>5.1f}%]"
        )

    return lines


# =====================================================================
# 3. 残基配对 vs 未配对
# =====================================================================
def matched_vs_unmatched():
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("3. 残基配对 vs 未配对（Fisher 检验）")
    lines.append("=" * 70)

    sc_rows = read_csv("results/sidechain_rmsd.csv")
    if not sc_rows:
        lines.append("  ⚠ 缺 sidechain_rmsd.csv")
        return lines

    matched_hit = matched_miss = 0
    unmatched_hit = unmatched_miss = 0
    for r in sc_rows:
        hit = fl(r.get("hit_4A"))
        if hit is None:
            continue
        sc = r.get("sidechain_rmsd_A", "")
        if sc not in ("", None):
            if int(hit) == 1:
                matched_hit += 1
            else:
                matched_miss += 1
        else:
            if int(hit) == 1:
                unmatched_hit += 1
            else:
                unmatched_miss += 1

    mn = matched_hit + matched_miss
    un = unmatched_hit + unmatched_miss
    mr = matched_hit / mn * 100 if mn > 0 else 0.0
    ur = unmatched_hit / un * 100 if un > 0 else 0.0

    lines.append(f"\n  Matched:    n={mn}  hit={matched_hit}  rate={mr:.2f}%")
    lines.append(f"  Unmatched:  n={un}  hit={unmatched_hit}  rate={ur:.2f}%")

    if mn > 0 and un > 0:
        odds, p = fisher_exact(
            [[matched_hit, matched_miss], [unmatched_hit, unmatched_miss]])
        lines.append(f"\n  Fisher exact:  p = {p:.6g}  odds ratio = {odds:.3f}")

    return lines


# =====================================================================
# 4. PDB 发布日期自查
# =====================================================================
def fetch_pdb_date(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("rcsb_accession_info", {}).get(
            "initial_release_date", "")
    except Exception:
        return None


def pdb_leakage_check(rows, out_dir, skip=False):
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("4. PDB 发布日期自查（AF3 训练集泄漏）")
    lines.append("=" * 70)

    pdb_ids = set()
    for r in rows:
        case = r.get("case", "")
        if "_" in case:
            pid = case.split("_")[0]
            if len(pid) == 4:
                pdb_ids.add(pid.upper())

    pdb_ids = sorted(pdb_ids)
    lines.append(f"\n  共 {len(pdb_ids)} 个唯一 PDB ID")

    if skip or not HAS_REQUESTS:
        lines.append("  (跳过联网查询；如需查询请 pip install requests)")
        return lines

    lines.append(f"  正在从 RCSB 查询（约 {len(pdb_ids)} 次请求）...")

    cache_path = Path(out_dir) / "pdb_release_dates.csv"
    cache = {}
    if cache_path.is_file():
        for r in read_csv(cache_path):
            cache[r["pdb_id"]] = r["release_date"]
        lines.append(f"  缓存已加载 {len(cache)} 条")

    results = []
    for i, pid in enumerate(pdb_ids, 1):
        if pid in cache:
            d = cache[pid]
        else:
            d = fetch_pdb_date(pid) or ""
            time.sleep(0.15)
        results.append({"pdb_id": pid, "release_date": d})
        if i % 20 == 0:
            lines.append(f"    ... {i}/{len(pdb_ids)}")

    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id", "release_date"])
        w.writeheader()
        w.writerows(results)
    lines.append(f"  ✅ {cache_path}")

    cutoff = "2021-09-30"
    dates = [r["release_date"] for r in results if r["release_date"]]
    if dates:
        dates.sort()
        n_before = sum(1 for d in dates if d < cutoff)
        lines.append(f"\n  AF3 训练截止假设: {cutoff}")
        lines.append(f"  发布日期 < 截止:  {n_before}")
        lines.append(f"  发布日期 ≥ 截止:  {len(dates) - n_before}")
        lines.append(f"  最早: {dates[0]}")
        lines.append(f"  最晚: {dates[-1]}")
        if n_before == 0:
            lines.append("\n  ✅ 无潜在训练集泄漏")
        else:
            lines.append(f"\n  ⚠ {n_before} 个结构可能早于截止，需人工核查")
    return lines


# =====================================================================
# 主流程
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--af3_csv", default="results/af3_model_summary.csv")
    p.add_argument("--eval_csv", default="results/evaluation.csv")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--skip-pdb", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    af3 = {r["case"]: r for r in read_csv(args.af3_csv)
           if r.get("model") == "af3_top1"}
    ev = {r["case"]: r for r in read_csv(args.eval_csv)}

    rows = []
    for case, r in af3.items():
        row = dict(r)
        if case in ev and "af3top1_rankfusion_err" in ev[case]:
            try:
                e = float(ev[case]["af3top1_rankfusion_err"])
                row["box_err"] = e
                row["hit_4A"] = 1 if e < 4.0 else 0
            except (ValueError, TypeError):
                pass
        rows.append(row)

    print(f"共 {len(rows)} 个 case\n")

    lines = []
    lines.extend(fisher_tests(rows))
    lines.extend(ci_report(rows))
    lines.extend(matched_vs_unmatched())
    lines.extend(pdb_leakage_check(rows, out_dir, skip=args.skip_pdb))

    text = "\n".join(lines)
    print(text)
    rp = out_dir / "supplementary_stats.txt"
    with open(rp, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n✅ {rp}")


if __name__ == "__main__":
    main()