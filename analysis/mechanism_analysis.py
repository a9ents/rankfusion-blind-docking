#!/usr/bin/env python3
"""mechanism_analysis.py — 融合机制分析（fpocket + P2Rank）。"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT / "src"))
import af3_common as ac

# 用相对路径（从 CWD 解析），与你诊断脚本一致
P2RANK_CSV = Path("results/p2rank_pocket_data.csv")
RESULTS_DIR = Path("results")
LIGAND_DIR = Path("DATA/ligands")
OUT_DIR = Path("results")

print(f"[info] CWD        = {Path.cwd()}")
print(f"[info] P2RANK_CSV = {P2RANK_CSV.resolve()}")
print(f"[info] LIGAND_DIR = {LIGAND_DIR.resolve()}")
print(f"[info] exists     = {LIGAND_DIR.is_dir()}")
if not LIGAND_DIR.is_dir():
    sys.exit("ERROR: LIGAND_DIR not found")

if not P2RANK_CSV.is_file():
    sys.exit(f"ERROR: {P2RANK_CSV} not found")


def rank_of(values, descending=False):
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i], reverse=descending)
    ranks = [0] * n
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def analyze_pockets(pockets, case, detector_name):
    valid = [p for p in pockets if p["energy"] is not None]
    if len(valid) < 2:
        return None

    scores = [p["score"] for p in valid]
    energies = [p["energy"] for p in valid]
    errs = [p["err"] for p in valid]

    true_idx = int(np.argmin(errs))
    rank_det = rank_of(scores, descending=True)
    rank_energy = rank_of(energies, descending=False)
    combined = [rank_energy[i] + 0.75 * rank_det[i] for i in range(len(valid))]
    rank_fusion = rank_of(combined, descending=False)

    if len(set(scores)) < 2 or len(set(energies)) < 2:
        rho = 0.0
    else:
        rho, _ = spearmanr(scores, energies)
        if np.isnan(rho):
            rho = 0.0
    rho = abs(rho)

    selected_det = valid[int(np.argmin(rank_det))]
    selected_energy = valid[int(np.argmin(rank_energy))]
    selected_fusion = valid[int(np.argmin(rank_fusion))]

    min_rank = min(rank_det[true_idx], rank_energy[true_idx])
    if rank_fusion[true_idx] < min_rank:
        synergy, equal, worse = 1, 0, 0
    elif rank_fusion[true_idx] == min_rank:
        synergy, equal, worse = 0, 1, 0
    else:
        synergy, equal, worse = 0, 0, 1

    return {
        "case": case,
        "detector": detector_name,
        "n_pockets": len(valid),
        "rho_det_energy": round(rho, 3),
        "rank_true_det": rank_det[true_idx],
        "rank_true_energy": rank_energy[true_idx],
        "rank_true_fusion": rank_fusion[true_idx],
        "synergy": synergy,
        "equal_best": equal,
        "worse_best": worse,
        "err_top1": selected_det["err"],
        "err_energy": selected_energy["err"],
        "err_fusion": selected_fusion["err"],
        "hit_top1": 1 if selected_det["err"] < 4 else 0,
        "hit_energy": 1 if selected_energy["err"] < 4 else 0,
        "hit_fusion": 1 if selected_fusion["err"] < 4 else 0,
    }


def find_ligand(case):
    ld = LIGAND_DIR / case
    if not ld.is_dir():
        return None
    for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb", "ligand.mol2"):
        p = ld / nm
        if p.is_file():
            return p
    return None


# ==================== P2Rank ====================
print("\n[1/2] 解析 P2Rank 数据...")
by_case_p2 = defaultdict(list)
with open(P2RANK_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if not r["vina_energy"]:
            continue
        by_case_p2[r["case"]].append({
            "score": float(r["p2rank_score"]),
            "energy": float(r["vina_energy"]),
            "err": float(r["err_to_true"]),
        })

rows = []
for case, pockets in by_case_p2.items():
    res = analyze_pockets(pockets, case, "P2Rank")
    if res:
        rows.append(res)
print(f"      P2Rank: {len(rows)} 个 case")

# ==================== fpocket ====================
print("[2/2] 解析 fpocket 数据...")
n_ok = 0
n_no_json = 0
n_no_lig = 0
n_small = 0
example_no_lig = []

for case_dir in sorted(RESULTS_DIR.iterdir()):
    if not case_dir.is_dir():
        continue
    case = case_dir.name
    json_path = case_dir / "exp_structure" / "pocket_ranks.json"
    if not json_path.is_file():
        n_no_json += 1
        continue

    lig = find_ligand(case)
    if lig is None:
        n_no_lig += 1
        if len(example_no_lig) < 3:
            ld = LIGAND_DIR / case
            example_no_lig.append((case, ld.is_dir(), str(ld)))
        continue
    true_c = ac.ligand_centroid(lig)
    if true_c is None:
        n_no_lig += 1
        continue

    try:
        with open(json_path, encoding="utf-8") as f:
            p2 = json.load(f)
    except Exception:
        n_no_json += 1
        continue

    pockets = []
    for pk in p2:
        c = pk.get("center")
        if not c or len(c) != 3:
            continue
        e = pk.get("vina_energy")
        if e is None:
            continue
        try:
            err = float(np.linalg.norm(np.array(c, dtype=float) - true_c))
        except Exception:
            continue
        pockets.append({
            "score": float(pk.get("druggability", 0)),
            "energy": float(e),
            "err": err,
        })

    if len(pockets) < 2:
        n_small += 1
        continue

    res = analyze_pockets(pockets, case, "fpocket")
    if res:
        rows.append(res)
        n_ok += 1

print(f"      fpocket: {n_ok} 个 case")
print(f"      跳过（无 json）: {n_no_json}")
print(f"      跳过（无配体）: {n_no_lig}")
print(f"      跳过（pocket < 2）: {n_small}")

if example_no_lig:
    print("      无配体示例:")
    for case, dir_exists, ld in example_no_lig:
        print(f"        {case}  dir_exists={dir_exists}  {ld}")
    # 打印一个实际存在的目录看看
    existing = [d for d in LIGAND_DIR.iterdir() if d.is_dir()][:3]
    print(f"      LIGAND_DIR 实际前 3 个子目录: {[d.name for d in existing]}")

# ==================== 写 CSV ====================
if rows:
    keys = list(rows[0].keys())
    with open(OUT_DIR / "mechanism_cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] mechanism_cases.csv ({len(rows)} 行)")

# ==================== 报告 ====================
report = []
report.append("=" * 78)
report.append("RankFusion 机制分析报告")
report.append("=" * 78)

report.append("\n表 1  两个检测器上的机制指标汇总")
report.append("-" * 78)
report.append(f"{'检测器':<10}{'n':>6}{'mean |rho|':>12}{'med |rho|':>12}"
              f"{'<4A(top1)':>12}{'<4A(E)':>10}{'<4A(F)':>10}")
for det in ["fpocket", "P2Rank"]:
    sub = [r for r in rows if r["detector"] == det]
    if not sub:
        continue
    rho = np.array([r["rho_det_energy"] for r in sub])
    h1 = np.mean([r["hit_top1"] for r in sub]) * 100
    he = np.mean([r["hit_energy"] for r in sub]) * 100
    hf = np.mean([r["hit_fusion"] for r in sub]) * 100
    report.append(f"{det:<10}{len(sub):>6}{rho.mean():>12.3f}"
                  f"{np.median(rho):>12.3f}{h1:>11.2f}%{he:>9.2f}%{hf:>9.2f}%")

report.append("\n表 2  融合相对两个单一信号最优值的排名变化")
report.append("-" * 78)
report.append(f"{'检测器':<10}{'n':>6}{'synergy':>12}{'equal':>12}{'worse':>12}")
for det in ["fpocket", "P2Rank"]:
    sub = [r for r in rows if r["detector"] == det]
    if not sub:
        continue
    syn = np.mean([r["synergy"] for r in sub]) * 100
    eq = np.mean([r["equal_best"] for r in sub]) * 100
    wor = np.mean([r["worse_best"] for r in sub]) * 100
    report.append(f"{det:<10}{len(sub):>6}{syn:>11.1f}%{eq:>11.1f}%{wor:>11.1f}%")

report.append("\n表 3  按 per-case 冗余度 |rho| 分组的 energy->fusion 误差降低")
report.append("-" * 78)
report.append(f"{'检测器':<10}{'组别':<10}{'n':>6}{'mean |rho|':>12}"
              f"{'d_err (A)':>12}{'d_hit (pp)':>12}")
for det in ["fpocket", "P2Rank"]:
    sub = [r for r in rows if r["detector"] == det]
    if not sub:
        continue
    med = np.median([r["rho_det_energy"] for r in sub])
    for name, group in [("low", [r for r in sub if r["rho_det_energy"] <= med]),
                        ("high", [r for r in sub if r["rho_det_energy"] > med])]:
        if not group:
            continue
        rho_m = np.mean([r["rho_det_energy"] for r in group])
        d_err = np.mean([r["err_energy"] - r["err_fusion"] for r in group])
        d_hit = (np.mean([r["hit_fusion"] for r in group]) -
                 np.mean([r["hit_energy"] for r in group])) * 100
        report.append(f"{det:<10}{name:<10}{len(group):>6}{rho_m:>12.3f}"
                      f"{d_err:>12.2f}{d_hit:>12.2f}")

report.append("\n" + "=" * 78)
report.append("关键结论")
report.append("=" * 78)
for det in ["fpocket", "P2Rank"]:
    sub = [r for r in rows if r["detector"] == det]
    if not sub:
        continue
    rho_med = np.median([r["rho_det_energy"] for r in sub])
    syn = np.mean([r["synergy"] for r in sub]) * 100
    eq = np.mean([r["equal_best"] for r in sub]) * 100
    d_he = (np.mean([r["hit_energy"] for r in sub]) -
            np.mean([r["hit_top1"] for r in sub])) * 100
    d_fh = (np.mean([r["hit_fusion"] for r in sub]) -
            np.mean([r["hit_energy"] for r in sub])) * 100
    report.append(f"\n[{det}]")
    report.append(f"  per-case 冗余度中位数: {rho_med:.3f}")
    report.append(f"  融合相对最优: synergy {syn:.1f}% / equal {eq:.1f}% / worse {100-syn-eq:.1f}%")
    report.append(f"  top1->energy 提升:    {d_he:+.2f} pp")
    report.append(f"  energy->fusion 额外:  {d_fh:+.2f} pp")

text = "\n".join(report)
print("\n" + text)
with open(OUT_DIR / "mechanism_report.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(f"\n[OK] mechanism_report.txt")