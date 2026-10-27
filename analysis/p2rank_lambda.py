#!/usr/bin/env python3
"""从 p2rank_pocket_data.csv 重算 P2Rank λ 敏感性。"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def rank_of(values, descending=False):
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i], reverse=descending)
    ranks = [0] * n
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


by_case = defaultdict(list)
with open("results/p2rank_pocket_data.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if not r["vina_energy"]:
            continue
        by_case[r["case"]].append({
            "score": float(r["p2rank_score"]),
            "energy": float(r["vina_energy"]),
            "err": float(r["err_to_true"]),
        })

print(f"共 {len(by_case)} 个 case\n")
print(f"{'lambda':<8}{'n':>6}{'mean':>10}{'median':>10}{'<4A':>10}")
rows = []
for lam in [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    errs = []
    for case, pockets in by_case.items():
        re = rank_of([p["energy"] for p in pockets], False)
        rs = rank_of([p["score"] for p in pockets], True)
        combined = [re[i] + lam * rs[i] for i in range(len(pockets))]
        errs.append(pockets[int(np.argmin(combined))]["err"])
    arr = np.array(errs)
    rows.append({"lambda": lam, "n": len(arr),
                 "mean": round(float(arr.mean()), 3),
                 "median": round(float(np.median(arr)), 3),
                 "lt_4A": round(float((arr < 4).mean()), 4)})
    print(f"{lam:<8.2f}{len(arr):>6}{arr.mean():>10.2f}"
          f"{np.median(arr):>10.2f}{(arr<4).mean()*100:>9.2f}%")

with open("results/lambda_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["lambda", "n", "mean", "median", "lt_4A"])
    w.writeheader()
    w.writerows(rows)
print(f"\n[OK] results/lambda_sensitivity.csv")