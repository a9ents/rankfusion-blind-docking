#!/usr/bin/env python3
"""
sample_consensus.py — AF3 5 个采样的配体位置一致性分析

对每个 case：
  1. 从 5 个 SEED-1_SAMPLE-X 的 CIF 提取配体质心
  2. 计算 5 个质心的离散度（平均成对距离、最大成对距离）
  3. 与 ipTM、配体定位误差对比

输出：results_sample_consensus.csv
"""
import csv
from pathlib import Path
from itertools import combinations

import numpy as np
import gemmi

AF3_DIR = Path("AF3_workflow")
FIXLIG_DIR = Path("fixligand_results")
OUT_CSV = Path("results_sample_consensus.csv")

aa3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
       "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
nt3 = {"A","C","G","U","DA","DC","DG","DT"}


def extract_ligand_centroid(cif_path):
    """从 CIF 提取配体质心。返回 np.array 或 None。"""
    try:
        st = gemmi.read_structure(str(cif_path))
        st.setup_entities()
        st.remove_hydrogens()
        st.remove_waters()
    except Exception:
        return None

    coords = []
    for chain in st[0]:
        is_prot = any(
            (r.name.strip().upper() in aa3) or (r.name.strip().upper() in nt3)
            for r in chain
        )
        if is_prot:
            continue
        for res in chain:
            for atom in res:
                if atom.element.name in ("H", "D"):
                    continue
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])

    if not coords:
        return None
    return np.mean(coords, axis=0)


def compute_consensus(centroids):
    """给定 N 个质心，返回离散度指标"""
    if len(centroids) < 2:
        return None
    C = np.array(centroids)
    n = len(C)
    d = []
    for i, j in combinations(range(n), 2):
        d.append(np.linalg.norm(C[i] - C[j]))
    d = np.array(d)
    return {
        "n_samples": n,
        "pair_mean": float(d.mean()),
        "pair_max": float(d.max()),
        "pair_min": float(d.min()),
        "std_x": float(C[:, 0].std()),
        "std_y": float(C[:, 1].std()),
        "std_z": float(C[:, 2].std()),
        "std_total": float(np.sqrt((C.std(axis=0) ** 2).sum())),
    }


def read_ranking_scores(case_dir):
    csv_path = list(case_dir.glob("*_ranking_scores.csv"))
    if not csv_path:
        return None, None
    scores = []
    try:
        with open(csv_path[0], encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    scores.append(float(r["ranking_score"]))
                except (ValueError, KeyError):
                    continue
    except OSError:
        return None, None
    if not scores:
        return None, None
    return float(np.mean(scores)), float(np.std(scores))


def process_case(case):
    case_dir = AF3_DIR / case
    if not case_dir.is_dir():
        return None

    # 找 5 个采样目录
    sample_dirs = sorted(case_dir.glob("SEED-1_SAMPLE-*"))
    centroids = []
    for sd in sample_dirs:
        cifs = sorted(sd.glob("*.cif"))
        if not cifs:
            continue
        c = extract_ligand_centroid(cifs[0])
        if c is not None:
            centroids.append(c)

    if len(centroids) < 2:
        return None

    cons = compute_consensus(centroids)
    if cons is None:
        return None

    # 读 ranking_score 分布
    r_mean, r_std = read_ranking_scores(case_dir)

    # 读 fixligand 的配体误差
    fix_csv = FIXLIG_DIR / "af3ligand_fixed_summary.csv"
    lig_err = None
    if fix_csv.is_file():
        with open(fix_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["case"] == case and row["err_A"]:
                    try:
                        lig_err = float(row["err_A"])
                    except ValueError:
                        pass
                    break

    return {
        "case": case,
        "n_samples": cons["n_samples"],
        "pair_mean_A": round(cons["pair_mean"], 2),
        "pair_max_A": round(cons["pair_max"], 2),
        "std_total_A": round(cons["std_total"], 2),
        "ranking_mean": round(r_mean, 3) if r_mean else "",
        "ranking_std": round(r_std, 3) if r_std else "",
        "ligand_err_A": round(lig_err, 2) if lig_err is not None else "",
    }


def main():
    cases = sorted([d.name for d in AF3_DIR.iterdir() if d.is_dir()])
    print(f"共 {len(cases)} 个 case\n")

    rows = []
    for i, case in enumerate(cases, 1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{len(cases)}] {case}")
        try:
            r = process_case(case)
        except Exception as e:
            print(f"    ⚠ {case}: {e}")
            r = None
        if r:
            rows.append(r)

    if not rows:
        print("⚠ 无结果")
        return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n✅ {OUT_CSV}  ({len(rows)} cases)")

    # 统计
    print("\n=== 采样一致性 vs 配体误差 ===")
    valid = [r for r in rows if r["ligand_err_A"] != ""]
    if not valid:
        return

    # 按 pair_mean 分档
    pair_means = np.array([r["pair_mean_A"] for r in valid])
    errs = np.array([r["ligand_err_A"] for r in valid])

    bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 100)]
    print(f"{'pair_mean':<14}{'n':>5}{'配体误差中位数':>14}{'<2Å':>8}{'<4Å':>8}")
    for lo, hi in bins:
        mask = (pair_means >= lo) & (pair_means < hi)
        if mask.sum() == 0:
            continue
        sub = errs[mask]
        print(f"[{lo},{hi})        {mask.sum():>5}{np.median(sub):>12.2f} Å"
              f"{(sub < 2).mean()*100:>7.1f}%{(sub < 4).mean()*100:>7.1f}%")

    # 相关性
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(pair_means, errs)
        print(f"\nSpearman(pair_mean, ligand_err) = {rho:+.3f}  p = {p:.2e}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()