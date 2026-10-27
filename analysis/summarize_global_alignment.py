#!/usr/bin/env python3
"""
summarize_global_alignment.py — 汇总 AF3 全局对齐 Rank Fusion 结果

读取 af3_postprocess_v2.py 生成的盒子文件：
  results/af3top1_rankfusion_{case}.box   （AF3 top1 全局对齐 + Rank Fusion）
  results/af3oracle_rankfusion_{case}.box （AF3 RMSD-oracle 全局对齐 + Rank Fusion）
  results/af3ligand_{case}.box            （AF3 共折叠配体 baseline）

与实验真值配体质心比较，输出误差和命中率。

用法：
    python summarize_global_alignment.py \
        --results_dir results \
        --ligand_dir ligands \
        --af3_dir AF3_workflow \
        --out_csv global_alignment_summary.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np


# =====================================================================
# 工具函数
# =====================================================================
def read_box_center(box_path):
    """读 Vina 盒子文件，返回中心坐标。"""
    p = Path(box_path)
    if not p.is_file():
        return None
    with open(p) as f:
        txt = f.read()
    vals = {}
    for line in txt.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip().lower()
            try:
                vals[k] = float(v.strip())
            except ValueError:
                pass
    if all(k in vals for k in ("center_x", "center_y", "center_z")):
        return np.array([vals["center_x"], vals["center_y"], vals["center_z"]])
    nums = []
    for tok in txt.replace("=", " ").split():
        try:
            nums.append(float(tok))
        except ValueError:
            pass
    if len(nums) >= 3:
        return np.array(nums[:3])
    return None


def ligand_centroid(path):
    """读 SDF/MOL/PDB 配体，返回重原子质心。"""
    p = Path(path)
    suf = p.suffix.lower()

    if suf in (".sdf", ".mol"):
        coords = []
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            if len(lines) < 4:
                return None
            n = int(lines[3][0:3].strip())
            for i in range(4, 4 + n):
                if i >= len(lines):
                    break
                line = lines[i]
                if len(line) < 30:
                    continue
                coords.append([float(line[0:10]), float(line[10:20]),
                               float(line[20:30])])
        except Exception:
            return None
        return np.mean(coords, axis=0) if coords else None

    if suf in (".pdb", ".ent"):
        coords = []
        with open(p) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    elem = line[76:78].strip().upper() if len(line) > 76 else ""
                    if elem in ("H", "D"):
                        continue
                    try:
                        coords.append([float(line[30:38]), float(line[38:46]),
                                       float(line[46:54])])
                    except ValueError:
                        continue
        return np.mean(coords, axis=0) if coords else None

    return None


def find_ligand(lig_case_dir):
    """在 ligands/{case}/ 下找配体文件。"""
    for name in ("ligand.sdf", "ligand.mol", "ligand.pdb",
                 "ligand.mol2", "ligand.ent"):
        p = lig_case_dir / name
        if p.is_file():
            return p
    return None


def err_to_true(box_center, true_center):
    if box_center is None or true_center is None:
        return None
    return float(np.linalg.norm(box_center - true_center))


# =====================================================================
# 统计
# =====================================================================
def print_stats(rows, label, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        print(f"\n[{label}] 无有效数据")
        return
    arr = np.array(vals)
    print(f"\n=== {label} ===")
    print(f"  n      = {len(arr)}")
    print(f"  mean   = {arr.mean():.2f} Å")
    print(f"  median = {np.median(arr):.2f} Å")
    print(f"  < 2 Å  = {(arr < 2).sum()}  ({(arr < 2).mean()*100:.2f}%)")
    print(f"  < 4 Å  = {(arr < 4).sum()}  ({(arr < 4).mean()*100:.2f}%)")


# =====================================================================
# 主流程
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results",
                   help="af3_postprocess_v2.py 的 --out_dir，存放 *.box")
    p.add_argument("--ligand_dir", default="ligands",
                   help="实验真值配体目录：ligands/{case}/ligand.sdf")
    p.add_argument("--af3_dir", default="AF3_workflow",
                   help="AF3 输出目录，用于枚举 case")
    p.add_argument("--cases", default=None,
                   help="逗号分隔的 case 列表，默认用 AF3 目录下所有子目录")
    p.add_argument("--out_csv", default="global_alignment_summary.csv")
    p.add_argument("--include_af3ligand", action="store_true",
                   help="是否把 AF3 共折叠配体 baseline 也纳入统计（注意坐标系不同）")
    args = p.parse_args()

    # 枚举 case
    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        root = Path(args.af3_dir)
        cases = sorted([d.name for d in root.iterdir() if d.is_dir()]) \
            if root.is_dir() else []

    print(f"共 {len(cases)} 个 case")
    print(f"结果目录: {Path(args.results_dir).resolve()}")
    print(f"配体目录: {Path(args.ligand_dir).resolve()}")

    rows = []
    for case in cases:
        r = {"case": case}

        lig_file = find_ligand(Path(args.ligand_dir) / case)
        if lig_file is None:
            r["status"] = "no_ligand"
            rows.append(r)
            continue

        true_center = ligand_centroid(lig_file)
        if true_center is None:
            r["status"] = "no_true_center"
            rows.append(r)
            continue

        r["true_x"] = round(true_center[0], 3)
        r["true_y"] = round(true_center[1], 3)
        r["true_z"] = round(true_center[2], 3)

        # 三个来源的盒子
        box_paths = {
            "af3top1":   Path(args.results_dir) / f"af3top1_rankfusion_{case}.box",
            "af3oracle": Path(args.results_dir) / f"af3oracle_rankfusion_{case}.box",
        }
        if args.include_af3ligand:
            box_paths["af3ligand"] = Path(args.results_dir) / f"af3ligand_{case}.box"

        any_ok = False
        for tag, bp in box_paths.items():
            c = read_box_center(bp)
            if c is None:
                r[f"{tag}_err_A"] = None
                r[f"{tag}_hit_4A"] = None
                r[f"{tag}_hit_2A"] = None
                continue
            err = err_to_true(c, true_center)
            r[f"{tag}_err_A"] = round(err, 3)
            r[f"{tag}_hit_4A"] = 1 if err < 4.0 else 0
            r[f"{tag}_hit_2A"] = 1 if err < 2.0 else 0
            any_ok = True

        r["status"] = "ok" if any_ok else "no_box"
        rows.append(r)

    # 写 CSV
    if rows:
        all_keys = ["case", "status"]
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n✅ {args.out_csv}")

    # 统计
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\n=== 结果 ===")
    print(f"  成功: {len(ok)}/{len(rows)}")

    print_stats(rows, "AF3 top1 (全局对齐) + Rank Fusion", "af3top1_err_A")
    print_stats(rows, "AF3 RMSD-oracle (全局对齐) + Rank Fusion", "af3oracle_err_A")
    if args.include_af3ligand:
        print_stats(rows, "AF3 共折叠配体 baseline (注意：坐标系与实验不同，仅供参考)", "af3ligand_err_A")


if __name__ == "__main__":
    main()