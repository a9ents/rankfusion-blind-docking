#!/usr/bin/env python3
"""从 p2rank_pocket_data.csv 生成 p2rank_energy（λ=0，纯 Vina）的 box。"""
import csv
from collections import defaultdict
from pathlib import Path

by_case = defaultdict(list)
with open("results/p2rank_pocket_data.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if not r["vina_energy"]:
            continue
        by_case[r["case"]].append({
            "energy": float(r["vina_energy"]),
            "cx": float(r["center_x"]),
            "cy": float(r["center_y"]),
            "cz": float(r["center_z"]),
        })

out = Path("results")
n = 0
for case, pockets in by_case.items():
    best = min(pockets, key=lambda p: p["energy"])
    with open(out / f"p2rank_energy_{case}.box", "w") as f:
        f.write(f"center_x = {best['cx']:.3f}\n")
        f.write(f"center_y = {best['cy']:.3f}\n")
        f.write(f"center_z = {best['cz']:.3f}\n")
    n += 1
print(f"生成 {n} 个 p2rank_energy box")