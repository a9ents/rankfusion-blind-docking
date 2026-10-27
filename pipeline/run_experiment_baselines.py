#!/usr/bin/env python3
"""
在实验结构上生成三个 baseline 盒子：
  - exp_top1        : λ=1000（等价于纯 druggability top1）
  - exp_energy      : λ=0（纯能量）
  - exp_rankfusion  : λ=0.75（本文方法）
"""
import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import auto_pipeline as ap


def run_case(case, exp_dir, ligand_dir, out_dir):
    exp_receptor = Path(exp_dir) / case / "receptor.pdb"
    lig_case_dir = Path(ligand_dir) / case
    if not exp_receptor.is_file() or not lig_case_dir.is_dir():
        print(f"  ✗ 跳过 {case}")
        return

    work = Path(out_dir) / case / "exp_structure"
    work.mkdir(parents=True, exist_ok=True)
    rec = work / "receptor.pdb"
    shutil.copy(exp_receptor, rec)

    ap.CFG["blind_mode"] = "pocket_guided"
    ap.CFG["pg_quick_exh"] = 4
    ap.CFG["pg_quick_modes"] = 3
    ap.CFG["exhaustiveness_fine"] = 32
    ap.CFG["num_modes_fine"] = 20

    for tag, lam in (("top1", 1000.0), ("energy", 0.0), ("rankfusion", 0.75)):
        ap.CFG["pg_rank_lambda"] = lam
        ok, _ = ap.run_pocket_guided(str(rec), str(lig_case_dir), str(work))
        box = work / "receptor.box"
        if ok and box.is_file():
            shutil.copy(box, Path(out_dir) / f"exp_{tag}_{case}.box")
            print(f"  ✅ exp_{tag}_{case}.box")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", default="exp_structures")
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--cases", default=None)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        cases = sorted([d.name for d in Path(args.exp_dir).iterdir() if d.is_dir()])

    print(f"共 {len(cases)} 个 case")
    for i, c in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {c}")
        run_case(c, args.exp_dir, args.ligand_dir, args.out_dir)


if __name__ == "__main__":
    main()