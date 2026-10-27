#!/usr/bin/env python3
"""诊断：全局 RMSD vs 配体误差"""
import csv
from pathlib import Path
import numpy as np

# 读配体误差
lig = {}
for r in csv.DictReader(open("fixligand_results/af3ligand_fixed_summary.csv")):
    if r["err_A"]:
        lig[r["case"]] = float(r["err_A"])

# 读 ipTM + 全局 RMSD
model = {}
for r in csv.DictReader(open("results/af3_model_summary.csv")):
    if r["model"] == "af3_top1":
        model[r["case"]] = {
            "ipTM": float(r["iptm"]),
            "rmsd": float(r["rmsd_to_exp"]),
        }

# 合并
rows = []
for c, e in lig.items():
    if c in model:
        rows.append((c, e, model[c]["ipTM"], model[c]["rmsd"]))

print(f"总 n = {len(rows)}\n")

# 按全局 RMSD 分档
print("=== 按全局 RMSD 分档 ===")
bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 20)]
print(f"{'RMSD':<12}{'n':>5}{'配体误差中位数':>16}{'<2Å':>8}{'<4Å':>8}")
for lo, hi in bins:
    sub = [r for r in rows if lo <= r[3] < hi]
    if not sub:
        continue
    errs = np.array([r[1] for r in sub])
    print(f"[{lo},{hi})     {len(sub):>5}{np.median(errs):>14.2f} Å"
          f"{(errs < 2).mean()*100:>7.1f}%{(errs < 4).mean()*100:>7.1f}%")

# 按 ipTM 分档
print("\n=== 按 ipTM 分档 ===")
bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
print(f"{'ipTM':<12}{'n':>5}{'配体误差中位数':>16}{'<2Å':>8}{'<4Å':>8}")
for lo, hi in bins:
    sub = [r for r in rows if lo <= r[2] <= hi]
    if not sub:
        continue
    errs = np.array([r[1] for r in sub])
    print(f"[{lo},{hi})     {len(sub):>5}{np.median(errs):>14.2f} Å"
          f"{(errs < 2).mean()*100:>7.1f}%{(errs < 4).mean()*100:>7.1f}%")

# 双重筛选
print("\n=== 双重筛选：ipTM ≥ 0.8 且 全局 RMSD < 3 Å ===")
sub = [r for r in rows if r[2] >= 0.8 and r[3] < 3]
if sub:
    errs = np.array([r[1] for r in sub])
    print(f"n = {len(sub)}")
    print(f"配体误差：mean={errs.mean():.2f} Å, median={np.median(errs):.2f} Å")
    print(f"< 4 Å: {(errs < 4).sum()} ({(errs < 4).mean()*100:.1f}%)")
    print(f"< 2 Å: {(errs < 2).sum()} ({(errs < 2).mean()*100:.1f}%)")
    print(f"\ncase 列表:")
    for c, e, i, r in sorted(sub, key=lambda x: x[1]):
        print(f"  {c}: err={e:.2f} Å, ipTM={i:.2f}, RMSD={r:.2f} Å")
else:
    print("无满足条件的 case")