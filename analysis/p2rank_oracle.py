#!/usr/bin/env python3
"""P2Rank 前 20 pocket 的 oracle 上限分析。"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT / "src"))
import af3_common as ac

P2RANK_OUT_ROOT = Path(r"E:\AF3环境\p2rank_work\p2rank_out")
MAX_POCKETS = 20


def load_top_pockets(case, root, n=MAX_POCKETS):
    candidates = list(Path(root).rglob(f"{case}.pdb_predictions.csv"))
    if not candidates:
        return []
    pockets = []
    with open(candidates[0], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v.strip() if isinstance(v, str) else v)
                   for k, v in row.items()}
            try:
                pockets.append({
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "center": np.array([float(row["center_x"]),
                                        float(row["center_y"]),
                                        float(row["center_z"])]),
                })
            except (KeyError, ValueError):
                continue
    pockets.sort(key=lambda p: p["rank"])
    return pockets[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--cases", default=None)
    args = p.parse_args()

    ligand_dir = Path(args.ligand_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        cases = sorted([d.name for d in ligand_dir.iterdir() if d.is_dir()])
    print(f"共 {len(cases)} 个 case")

    rows = []
    for i, case in enumerate(cases, 1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{len(cases)}] {case}")

        lig_file = None
        for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb", "ligand.mol2"):
            pth = ligand_dir / case / nm
            if pth.is_file():
                lig_file = pth
                break
        if lig_file is None:
            continue

        true_c = ac.ligand_centroid(lig_file)
        if true_c is None:
            continue

        pockets = load_top_pockets(case, P2RANK_OUT_ROOT)
        if not pockets:
            continue

        dists = [float(np.linalg.norm(p["center"] - true_c)) for p in pockets]
        best_idx = int(np.argmin(dists))
        top1_dist = dists[0]

        rows.append({
            "case": case,
            "n_pockets": len(pockets),
            "top1_err": round(top1_dist, 3),
            "oracle_err": round(dists[best_idx], 3),
            "oracle_rank": best_idx + 1,
        })

    csv_path = out_dir / "p2rank_oracle_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {csv_path}")

    top1 = np.array([r["top1_err"] for r in rows])
    oracle = np.array([r["oracle_err"] for r in rows])
    ranks = np.array([r["oracle_rank"] for r in rows])

    print(f"\n===== P2Rank oracle 分析（n={len(rows)}）=====")
    print(f"  top1    < 4Å: {(top1 < 4).sum()} ({(top1 < 4).mean()*100:.2f}%)")
    print(f"  oracle  < 4Å: {(oracle < 4).sum()} ({(oracle < 4).mean()*100:.2f}%)")
    print(f"  top1    平均误差: {top1.mean():.2f} Å, 中位: {np.median(top1):.2f} Å")
    print(f"  oracle  平均误差: {oracle.mean():.2f} Å, 中位: {np.median(oracle):.2f} Å")


if __name__ == "__main__":
    main()