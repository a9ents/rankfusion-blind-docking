#!/usr/bin/env python3
"""从 exp_structure/pocket_ranks.json 重算 fpocket λ 敏感性（424 个）。"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT / "src"))
import af3_common as ac


def rank_of(values, descending=False):
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i], reverse=descending)
    ranks = [0] * n
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--lambdas", default="0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    args = p.parse_args()

    lam_list = [float(x) for x in args.lambdas.split(",")]

    by_case = defaultdict(list)
    for case_dir in sorted(Path(args.results_dir).iterdir()):
        if not case_dir.is_dir():
            continue
        case = case_dir.name
        json_path = case_dir / "exp_structure" / "pocket_ranks.json"
        if not json_path.is_file():
            continue
        lig_dir = Path(args.ligand_dir) / case
        lig = None
        for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb", "ligand.mol2"):
            pth = lig_dir / nm
            if pth.is_file():
                lig = pth
                break
        if lig is None:
            continue
        true_c = ac.ligand_centroid(lig)
        if true_c is None:
            continue
        with open(json_path, encoding="utf-8") as f:
            pockets = json.load(f)
        for pk in pockets:
            c = pk.get("center")
            if not c or len(c) != 3:
                continue
            err = float(np.linalg.norm(np.array(c, dtype=float) - true_c))
            by_case[case].append({
                "druggability": float(pk.get("druggability", 0)),
                "vina_energy": pk.get("vina_energy"),
                "err": err,
            })

    print(f"共 {len(by_case)} 个 case")

    def eval_lam(lam):
        errs = []
        for case, pockets in by_case.items():
            valid = [p for p in pockets if p["vina_energy"] is not None]
            if not valid:
                continue
            re = rank_of([p["vina_energy"] for p in valid], False)
            rs = rank_of([p["druggability"] for p in valid], True)
            combined = [re[i] + lam * rs[i] for i in range(len(valid))]
            errs.append(valid[int(np.argmin(combined))]["err"])
        return np.array(errs)

    rows = []
    print(f"\n{'λ':<8}{'n':>6}{'mean':>10}{'median':>10}{'<4Å':>10}")
    for lam in lam_list:
        arr = eval_lam(lam)
        if len(arr) == 0:
            continue
        row = {"lambda": lam, "n": len(arr),
               "mean": round(float(arr.mean()), 3),
               "median": round(float(np.median(arr)), 3),
               "lt_4A": round(float((arr < 4).mean()), 4)}
        rows.append(row)
        print(f"{lam:<8.2f}{len(arr):>6}{arr.mean():>10.2f}"
              f"{np.median(arr):>10.2f}{(arr<4).mean()*100:>9.2f}%")

    out = Path(args.out_dir) / "fpocket_lambda_sensitivity.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lambda", "n", "mean", "median", "lt_4A"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()