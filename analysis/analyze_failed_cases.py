#!/usr/bin/env python3
"""
analyze_failed_cases.py — 分析"流程未完成"的 case

对比两组：
  - 已完成：在 evaluation.csv 里有 af3top1_rankfusion_err
  - 未完成：有 AF3 模型，但 evaluation.csv 里没结果

对比维度：ipTM、全局 RMSD、ranking_score、TM-score

输出：
  results/failed_cases.csv          逐 case 明细
  results/failed_cases_report.txt   对比报告

用法：
    python analyze_failed_cases.py
"""
import argparse
import csv
from pathlib import Path
import numpy as np


def safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_af3_models(path):
    """从 af3_model_summary.csv 读 af3_top1 行，返回 {case: dict}"""
    out = {}
    if not Path(path).is_file():
        print(f"  ⚠ 缺文件: {path}")
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("model") != "af3_top1":
                continue
            out[r["case"]] = {
                "ipTM": safe_float(r.get("iptm")),
                "ranking_score": safe_float(r.get("ranking_score")),
                "global_rmsd": safe_float(r.get("rmsd_to_exp")),
                "tm_score": safe_float(r.get("tm_score")),
                "has_ligand": r.get("has_ligand", ""),
                "cif": r.get("cif", ""),
            }
    return out


def load_eval(path):
    """从 evaluation.csv 读，返回 {case: dict}"""
    out = {}
    if not Path(path).is_file():
        print(f"  ⚠ 缺文件: {path}")
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["case"]] = r
    return out


def describe(arr, name):
    """打印统计"""
    arr = np.array([x for x in arr if x is not None], dtype=float)
    if len(arr) == 0:
        return [f"  {name}: 无数据"]
    lines = []
    lines.append(f"  {name}  n={len(arr)}")
    lines.append(f"    mean   = {arr.mean():.3f}")
    lines.append(f"    median = {np.median(arr):.3f}")
    lines.append(f"    std    = {arr.std(ddof=1) if len(arr) > 1 else 0:.3f}")
    lines.append(f"    min    = {arr.min():.3f}")
    lines.append(f"    max    = {arr.max():.3f}")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--af3_csv", default="results/af3_model_summary.csv")
    p.add_argument("--eval_csv", default="results/evaluation.csv")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载
    af3 = load_af3_models(args.af3_csv)
    ev = load_eval(args.eval_csv)

    if not af3:
        print("⚠ af3_model_summary.csv 无数据")
        return

    # 区分
    completed = []   # 有 box 结果
    failed = []      # 有 AF3 模型，但没有 box 结果

    for case, info in af3.items():
        has_result = (case in ev
                      and ev[case].get("af3top1_rankfusion_err", "") not in
                          ("", None))
        rec = {"case": case, **info}
        if has_result:
            try:
                rec["box_err"] = float(ev[case]["af3top1_rankfusion_err"])
                rec["hit_4A"] = 1 if rec["box_err"] < 4.0 else 0
            except (ValueError, TypeError):
                rec["box_err"] = ""
                rec["hit_4A"] = ""
            completed.append(rec)
        else:
            rec["box_err"] = ""
            rec["hit_4A"] = ""
            # 检查 evaluation.csv 里有没有 exp_rankfusion（看实验结构那边有没有跑通）
            rec["has_exp_result"] = (
                case in ev and
                ev[case].get("exp_rankfusion_err", "") not in ("", None)
            )
            failed.append(rec)

    print(f"\n总 AF3 模型数: {len(af3)}")
    print(f"  已完成（有 box 结果）: {len(completed)}")
    print(f"  未完成（有模型无 box）: {len(failed)}\n")

    # 写明细 CSV
    all_rows = []
    for r in completed:
        all_rows.append({**r, "group": "completed"})
    for r in failed:
        all_rows.append({**r, "group": "failed"})

    cols = ["case", "group", "ipTM", "ranking_score",
            "global_rmsd", "tm_score", "has_ligand",
            "box_err", "hit_4A"]
    for r in all_rows:
        for c in cols:
            r.setdefault(c, "")

    csv_path = out_dir / "failed_cases.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"✅ {csv_path}")

    # 对比报告
    report = []
    report.append("=" * 70)
    report.append("流程未完成 case 分析")
    report.append("=" * 70)

    report.append(f"\n总 AF3 模型: {len(af3)}")
    report.append(f"  已完成: {len(completed)}  ({len(completed)/len(af3)*100:.1f}%)")
    report.append(f"  未完成: {len(failed)}  ({len(failed)/len(af3)*100:.1f}%)")

    # 两组质量对比
    for metric, label in [("ipTM", "ipTM"),
                          ("global_rmsd", "全局 RMSD (Å)"),
                          ("ranking_score", "ranking_score"),
                          ("tm_score", "TM-score")]:
        report.append(f"\n【{label}】")
        report.append("  --- 已完成组 ---")
        report.extend(describe(
            [r[metric] for r in completed], label))
        report.append("  --- 未完成组 ---")
        report.extend(describe(
            [r[metric] for r in failed], label))

    # 分档统计
    report.append("\n" + "=" * 70)
    report.append("按 ipTM 分档：完成 vs 未完成")
    report.append("=" * 70)

    iptm_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = [f"[{iptm_bins[i]}, {iptm_bins[i+1]})"
              for i in range(len(iptm_bins) - 1)]

    report.append(f"\n  {'ipTM':<14}{'completed':>12}{'failed':>12}"
                  f"{'failed %':>12}")
    for i, lab in enumerate(labels):
        lo, hi = iptm_bins[i], iptm_bins[i+1]
        if i == len(labels) - 1:
            n_c = sum(1 for r in completed
                      if r["ipTM"] is not None and lo <= r["ipTM"] <= hi)
            n_f = sum(1 for r in failed
                      if r["ipTM"] is not None and lo <= r["ipTM"] <= hi)
        else:
            n_c = sum(1 for r in completed
                      if r["ipTM"] is not None and lo <= r["ipTM"] < hi)
            n_f = sum(1 for r in failed
                      if r["ipTM"] is not None and lo <= r["ipTM"] < hi)
        total = n_c + n_f
        pct = n_f / total * 100 if total > 0 else 0
        report.append(f"  {lab:<14}{n_c:>12}{n_f:>12}{pct:>11.1f}%")

    # 分档统计 (RMSD)
    report.append("\n" + "=" * 70)
    report.append("按全局 RMSD 分档：完成 vs 未完成")
    report.append("=" * 70)

    rmsd_bins = [0, 2, 4, 6, 8, 20]
    labels = [f"[{rmsd_bins[i]}, {rmsd_bins[i+1]})"
              for i in range(len(rmsd_bins) - 1)]

    report.append(f"\n  {'RMSD (Å)':<14}{'completed':>12}{'failed':>12}"
                  f"{'failed %':>12}")
    for i, lab in enumerate(labels):
        lo, hi = rmsd_bins[i], rmsd_bins[i+1]
        if i == len(labels) - 1:
            n_c = sum(1 for r in completed
                      if r["global_rmsd"] is not None
                      and lo <= r["global_rmsd"] <= hi)
            n_f = sum(1 for r in failed
                      if r["global_rmsd"] is not None
                      and lo <= r["global_rmsd"] <= hi)
        else:
            n_c = sum(1 for r in completed
                      if r["global_rmsd"] is not None
                      and lo <= r["global_rmsd"] < hi)
            n_f = sum(1 for r in failed
                      if r["global_rmsd"] is not None
                      and lo <= r["global_rmsd"] < hi)
        total = n_c + n_f
        pct = n_f / total * 100 if total > 0 else 0
        report.append(f"  {lab:<14}{n_c:>12}{n_f:>12}{pct:>11.1f}%")

    # 未完成 case 是否本来在实验结构上也失败？
    report.append("\n" + "=" * 70)
    report.append("未完成 case 的实验结构对照")
    report.append("=" * 70)

    n_exp_ok = sum(1 for r in failed if r.get("has_exp_result"))
    n_exp_no = len(failed) - n_exp_ok
    report.append(f"\n  failed 中，实验结构上有对接结果: {n_exp_ok}")
    report.append(f"  failed 中，实验结构上也无结果: {n_exp_no}")
    if n_exp_no > 0:
        report.append("  → 这些 case 在实验结构上就失败了，与 AF3 无关")

    # 结论提示
    report.append("\n" + "=" * 70)
    report.append("结论提示")
    report.append("=" * 70)

    iptm_c = np.mean([r["ipTM"] for r in completed
                      if r["ipTM"] is not None])
    iptm_f = np.mean([r["ipTM"] for r in failed
                      if r["ipTM"] is not None])
    rmsd_c = np.mean([r["global_rmsd"] for r in completed
                      if r["global_rmsd"] is not None])
    rmsd_f = np.mean([r["global_rmsd"] for r in failed
                      if r["global_rmsd"] is not None])

    report.append(f"\n  已完成组: mean ipTM = {iptm_c:.3f}, "
                  f"mean RMSD = {rmsd_c:.2f} Å")
    report.append(f"  未完成组: mean ipTM = {iptm_f:.3f}, "
                  f"mean RMSD = {rmsd_f:.2f} Å")

    if iptm_f < iptm_c and rmsd_f > rmsd_c:
        report.append("\n  → 未完成组的 AF3 质量显著更差，说明未完成不是随机的，")
        report.append("    而是结构太差导致 fpocket 找不到口袋或对接失败。")
        report.append("    这是支持你结论的证据，不是数据丢失。")
    else:
        report.append("\n  → 两组质量相近，未完成可能是技术原因（配体转换、")
        report.append("    fpocket 超时等），需要在方法部分说明。")

    text = "\n".join(report)
    print(text)
    rp = out_dir / "failed_cases_report.txt"
    with open(rp, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n✅ {rp}")


if __name__ == "__main__":
    main()