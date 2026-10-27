#!/usr/bin/env python3
"""从 pocket_ranks.json 重算 fpocket oracle（424 个复合物）。"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT / "src"))
import af3_common as ac


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_csv", default="results/fpocket_oracle_summary.csv")
    args = p.parse_args()

    lig_dir = Path(args.ligand_dir)
    res_dir = Path(args.results_dir)

    rows = []
    for case_dir in sorted(lig_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        case = case_dir.name

        # 真值
        lig = None
        for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb", "ligand.mol2"):
            pth = case_dir / nm
            if pth.is_file():
                lig = pth
                break
        if lig is None:
            continue
        true_c = ac.ligand_centroid(lig)
        if true_c is None:
            continue

        # pocket_ranks.json
        json_path = res_dir / case / "exp_structure" / "pocket_ranks.json"
        if not json_path.is_file():
            continue

        with open(json_path, encoding="utf-8") as f:
            pockets = json.load(f)
        if not pockets:
            continue

        dists = []
        for pk in pockets:
            c = pk.get("center")
            if c is None or len(c) != 3:
                continue
            c = np.array(c, dtype=float)
            dists.append(float(np.linalg.norm(c - true_c)))
        if not dists:
            continue

        rows.append({
            "case": case,
            "n_pockets": len(dists),
            "top1_err": round(dists[0], 3),
            "oracle_err": round(min(dists), 3),
        })

    out_path = Path(args.out_csv)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "n_pockets", "top1_err", "oracle_err"])
        w.writeheader()
        w.writerows(rows)

    top1 = np.array([r["top1_err"] for r in rows])
    oracle = np.array([r["oracle_err"] for r in rows])

    print(f"n = {len(rows)}")
    print(f"  top1   < 4Å: {(top1 < 4).sum()} ({(top1 < 4).mean()*100:.2f}%)")
    print(f"  oracle < 4Å: {(oracle < 4).sum()} ({(oracle < 4).mean()*100:.2f}%)")
    print(f"  top1   mean/median: {top1.mean():.2f} / {np.median(top1):.2f}")
    print(f"  oracle mean/median: {oracle.mean():.2f} / {np.median(oracle):.2f}")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()