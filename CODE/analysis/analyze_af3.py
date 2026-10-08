import argparse
import csv
from pathlib import Path

import numpy as np


# =====================================================================
# 加载
# =====================================================================
def load_model_summary(path):
    """只取 af3_top1 行，返回 list[dict]"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("model") != "af3_top1":
                continue
            if not r.get("rmsd_to_exp"):
                continue
            rows.append(r)
    return rows


def load_eval(path):
    """读 evaluation.csv，返回 {case: {...}}"""
    if not Path(path).is_file():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["case"]] = r
    return out


# =====================================================================
# 统计
# =====================================================================
def safe_float(v, default=None):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def basic_stats(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) == 0:
        return {}
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def binned_stats(values, bins, labels=None):
    """按 bins 分档，返回每档的 n / 占比"""
    values = np.array(values, dtype=float)
    if labels is None:
        labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)]
    out = []
    for i, lab in enumerate(labels):
        lo, hi = bins[i], bins[i + 1]
        if i == len(labels) - 1:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        n = int(mask.sum())
        out.append({
            "bin": lab,
            "n": n,
            "pct": round(n / len(values) * 100, 2) if len(values) else 0.0,
        })
    return out


# =====================================================================
# 合并成功率
# =====================================================================
def merge_with_eval(model_rows, eval_map, method="af3top1_rankfusion"):
    key = f"{method}_err"
    out = []
    for r in model_rows:
        case = r["case"]
        row = {
            "case": case,
            "iptm": safe_float(r.get("iptm")),
            "ptm": safe_float(r.get("ptm")),
            "ranking_score": safe_float(r.get("ranking_score")),
            "rmsd_to_exp": safe_float(r.get("rmsd_to_exp")),
            "tm_score": safe_float(r.get("tm_score")),
            "ligand_err": None,
            "hit_4A": None,
        }
        if case in eval_map and key in eval_map[case]:
            e = safe_float(eval_map[case][key])
            if e is not None:
                row["ligand_err"] = e
                row["hit_4A"] = 1 if e < 4.0 else 0
        out.append(row)
    return out


def hit_rate_by_bin(pairs, value_key, bins, labels=None):
    """按某个特征分档，统计 Rank Fusion 成功率"""
    ok = [p for p in pairs if p[value_key] is not None and p["hit_4A"] is not None]
    if not ok:
        return []

    if labels is None:
        labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)]

    out = []
    for i, lab in enumerate(labels):
        lo, hi = bins[i], bins[i + 1]
        if i == len(labels) - 1:
            sub = [p for p in ok if lo <= p[value_key] <= hi]
        else:
            sub = [p for p in ok if lo <= p[value_key] < hi]
        if not sub:
            out.append({
                "bin": lab, "n": 0,
                "mean_feature": "", "mean_ligand_err": "",
                "hit_rate_4A": "",
            })
            continue
        feats = [p[value_key] for p in sub]
        errs = [p["ligand_err"] for p in sub if p["ligand_err"] is not None]
        hits = [p["hit_4A"] for p in sub]
        out.append({
            "bin": lab,
            "n": len(sub),
            "mean_feature": round(float(np.mean(feats)), 3),
            "mean_ligand_err": round(float(np.mean(errs)), 3) if errs else "",
            "hit_rate_4A": round(sum(hits) / len(hits), 4),
        })
    return out


# =====================================================================
# 主流程
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_csv", default="results/af3_model_summary.csv")
    p.add_argument("--eval_csv", default="results/evaluation.csv")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载 ----
    rows = load_model_summary(args.summary_csv)
    if not rows:
        print(f"⚠ 没读到数据：{args.summary_csv}")
        return
    print(f"读取 {len(rows)} 个 AF3-top1 模型\n")

    iptm   = [safe_float(r["iptm"], 0.0) for r in rows]
    rmsd   = [safe_float(r["rmsd_to_exp"], 0.0) for r in rows]
    rank   = [safe_float(r["ranking_score"], 0.0) for r in rows]
    tm     = [safe_float(r["tm_score"], 0.0) for r in rows]

    # ---- 控制台报告 ----
    report = []
    report.append("=" * 70)
    report.append("AF3 模型质量分析")
    report.append("=" * 70)

    for name, arr in [("ipTM", iptm), ("ranking_score", rank),
                      ("TM-score", tm), ("RMSD to exp (Å)", rmsd)]:
        s = basic_stats(arr)
        report.append(f"\n【{name}】 n={s['n']}")
        report.append(f"  mean   = {s['mean']:.3f}")
        report.append(f"  median = {s['median']:.3f}")
        report.append(f"  std    = {s['std']:.3f}")
        report.append(f"  Q1–Q3  = {s['q25']:.3f} – {s['q75']:.3f}")
        report.append(f"  min–max = {s['min']:.3f} – {s['max']:.3f}")

    # ---- 分档 ----
    report.append("\n" + "=" * 70)
    report.append("分档统计")
    report.append("=" * 70)

    iptm_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rmsd_bins = [0, 2, 4, 6, 8, 20]

    report.append("\n【ipTM 分档】")
    for b in binned_stats(iptm, iptm_bins):
        report.append(f"  {b['bin']:<15} n={b['n']:>4}  ({b['pct']:>5.1f}%)")

    report.append("\n【RMSD 分档】")
    for b in binned_stats(rmsd, rmsd_bins):
        report.append(f"  {b['bin']:<15} n={b['n']:>4}  ({b['pct']:>5.1f}%)")

    # ---- 相关性 ----
    report.append("\n" + "=" * 70)
    report.append("相关性")
    report.append("=" * 70)
    try:
        from scipy.stats import spearmanr, pearsonr
        rho1, p1 = spearmanr(iptm, rmsd)
        rho2, p2 = pearsonr(iptm, rmsd)
        report.append(f"\nipTM vs RMSD:")
        report.append(f"  Spearman ρ = {rho1:+.3f}  p = {p1:.2e}")
        report.append(f"  Pearson  r = {rho2:+.3f}  p = {p2:.2e}")

        rho3, p3 = spearmanr(rank, rmsd)
        report.append(f"\nranking_score vs RMSD:")
        report.append(f"  Spearman ρ = {rho3:+.3f}  p = {p3:.2e}")
    except ImportError:
        report.append("  (需要 scipy: pip install scipy)")

    # ---- 合并 evaluation ----
    eval_map = load_eval(args.eval_csv)
    pairs = merge_with_eval(rows, eval_map)

    n_with_hit = sum(1 for p in pairs if p["hit_4A"] is not None)
    if n_with_hit > 0:
        report.append("\n" + "=" * 70)
        report.append(f"Rank Fusion 成功率 vs AF3 质量（n={n_with_hit} 个已评估）")
        report.append("=" * 70)

        # 总体
        hits = [p["hit_4A"] for p in pairs if p["hit_4A"] is not None]
        errs = [p["ligand_err"] for p in pairs if p["ligand_err"] is not None]
        report.append(f"\n总体: hit(<4Å) = {sum(hits)/len(hits)*100:.2f}%  "
                      f"mean_err = {np.mean(errs):.2f} Å")

        # 按 ipTM
        report.append("\n【按 ipTM 分档】")
        bins = hit_rate_by_bin(pairs, "iptm", iptm_bins)
        report.append(f"  {'bin':<15}{'n':>6}{'mean ipTM':>12}"
                      f"{'mean err':>12}{'hit_4Å':>10}")
        for b in bins:
            if b["n"] == 0:
                continue
            report.append(f"  {b['bin']:<15}{b['n']:>6}"
                          f"{b['mean_feature']:>12.3f}"
                          f"{b['mean_ligand_err']:>12.3f}"
                          f"{b['hit_rate_4A']*100:>9.2f}%")

        # 按 RMSD
        report.append("\n【按 RMSD 分档】")
        bins = hit_rate_by_bin(pairs, "rmsd_to_exp", rmsd_bins)
        report.append(f"  {'bin':<15}{'n':>6}{'mean RMSD':>12}"
                      f"{'mean err':>12}{'hit_4Å':>10}")
        for b in bins:
            if b["n"] == 0:
                continue
            report.append(f"  {b['bin']:<15}{b['n']:>6}"
                          f"{b['mean_feature']:>12.3f}"
                          f"{b['mean_ligand_err']:>12.3f}"
                          f"{b['hit_rate_4A']*100:>9.2f}%")

        # 高置信子集
        high = [p for p in pairs
                if p["iptm"] is not None and p["iptm"] >= 0.8
                and p["hit_4A"] is not None]
        if high:
            hits_h = [p["hit_4A"] for p in high]
            report.append(f"\n【ipTM ≥ 0.8 子集】 n={len(high)}  "
                          f"hit(<4Å) = {sum(hits_h)/len(hits_h)*100:.2f}%")
        low = [p for p in pairs
               if p["iptm"] is not None and p["iptm"] < 0.4
               and p["hit_4A"] is not None]
        if low:
            hits_l = [p["hit_4A"] for p in low]
            report.append(f"【ipTM < 0.4 子集】 n={len(low)}  "
                          f"hit(<4Å) = {sum(hits_l)/len(hits_l)*100:.2f}%")

        # 写 CSV
        pairs_csv = out_dir / "af3_analysis_pairs.csv"
        with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            w.writeheader()
            w.writerows(pairs)
        report.append(f"\n✅ {pairs_csv}")

        # 分档 CSV
        bins_csv = out_dir / "af3_analysis_bins.csv"
        with open(bins_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature", "bin", "n", "mean_feature",
                        "mean_ligand_err", "hit_rate_4A"])
            for b in hit_rate_by_bin(pairs, "iptm", iptm_bins):
                if b["n"] > 0:
                    w.writerow(["ipTM", b["bin"], b["n"],
                                b["mean_feature"], b["mean_ligand_err"],
                                b["hit_rate_4A"]])
            for b in hit_rate_by_bin(pairs, "rmsd_to_exp", rmsd_bins):
                if b["n"] > 0:
                    w.writerow(["RMSD", b["bin"], b["n"],
                                b["mean_feature"], b["mean_ligand_err"],
                                b["hit_rate_4A"]])
        report.append(f"✅ {bins_csv}")

    # ---- 打印 + 存报告 ----
    text = "\n".join(report)
    print(text)
    report_path = out_dir / "af3_analysis_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n✅ {report_path}")


if __name__ == "__main__":
    main()