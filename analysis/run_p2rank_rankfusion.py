#!/usr/bin/env python3
"""
run_p2rank_rankfusion.py — P2Rank + RankFusion
"""
import os
import argparse
import csv
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "pipeline"))

import auto_pipeline as ap          # type: ignore # noqa: E402

# ============================================================
# 配置（自适应，任何电脑都不用改）
# ============================================================
P2RANK_OUT_ROOT = Path(os.environ.get(
    "P2RANK_OUT_ROOT", str(_ROOT / "work" / "p2rank_out")))
WORK_ROOT = Path(os.environ.get(
    "P2RANK_RF_WORK_ROOT", str(_ROOT / "work" / "p2rank_rf")))

MAX_POCKETS = 20
LAMBDA = 0.75
BOX_SIZE = 25.0
QUICK_EXH = 4
QUICK_MODES = 3
SEED = 42


def load_p2rank_pockets(case, p2rank_root):
    root = Path(p2rank_root)
    candidates = list(root.rglob(f"{case}.pdb_predictions.csv"))
    if not candidates:
        return []
    pred_csv = candidates[0]

    pockets = []
    with open(pred_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            try:
                pockets.append({
                    "pocket": row["name"],
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "center": np.array([float(row["center_x"]), float(row["center_y"]), float(row["center_z"])]),
                })
            except (KeyError, ValueError):
                continue

    pockets.sort(key=lambda p: p["rank"])
    return pockets[:MAX_POCKETS]


def _dock_one(args):
    rec_pdbqt, lig_pdbqt, center, out_path, pocket_name = args
    cmd = [
        ap.VINA_BIN, "--receptor", rec_pdbqt, "--ligand", lig_pdbqt, "--out", out_path,
        "--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}", "--center_z", f"{center[2]:.3f}",
        "--size_x", str(BOX_SIZE), "--size_y", str(BOX_SIZE), "--size_z", str(BOX_SIZE),
        "--exhaustiveness", str(QUICK_EXH), "--num_modes", str(QUICK_MODES), "--seed", str(SEED),
    ]
    import subprocess
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return pocket_name, None
    return pocket_name, ap.read_best_energy(out_path)


def process_one(case, exp_dir, ligand_dir, out_dir):
    exp_rec = Path(exp_dir) / case / "receptor.pdb"
    lig_case = Path(ligand_dir) / case
    if not exp_rec.is_file() or not lig_case.is_dir():
        return case, None, "missing input"

    pockets = load_p2rank_pockets(case, P2RANK_OUT_ROOT)
    if not pockets:
        return case, None, "no p2rank output"

    work = WORK_ROOT / case
    work.mkdir(parents=True, exist_ok=True)

    rec_pdbqt = work / "receptor.pdbqt"
    if not rec_pdbqt.is_file() or rec_pdbqt.stat().st_size < 100:
        if not ap.pdb_to_pdbqt_receptor(str(exp_rec), str(rec_pdbqt)):
            return case, None, "receptor pdbqt failed"

    ligs = sorted(lig_case.glob("*.pdb"))
    if not ligs:
        return case, None, "no ligand pdb"

    lig_pdbqts = []
    for lig in ligs:
        out_pdbqt = work / f"{lig.stem}.pdbqt"
        if not out_pdbqt.is_file() or out_pdbqt.stat().st_size < 100:
            if not ap.pdb_to_pdbqt_ligand(str(lig), str(out_pdbqt)):
                continue
        lig_pdbqts.append(out_pdbqt)

    if not lig_pdbqts:
        return case, None, "no ligand pdbqt"

    def count_atoms(p):
        try:
            return sum(1 for l in open(p) if l.startswith(("ATOM", "HETATM")))
        except OSError:
            return 0
    lig_pdbqts.sort(key=count_atoms)
    probe = lig_pdbqts[0]

    tasks = []
    for p in pockets:
        out_pdbqt = work / f"quick_{p['pocket']}.pdbqt"
        tasks.append((str(rec_pdbqt), str(probe), p["center"], str(out_pdbqt), p["pocket"]))

    results = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_dock_one, t): t[4] for t in tasks}
        for fut in as_completed(futures):
            name, e = fut.result()
            results[name] = e

    valid = [p for p in pockets if results.get(p["pocket"]) is not None]
    if not valid:
        return case, None, "all docking failed"

    by_e = sorted(valid, key=lambda p: results[p["pocket"]])
    by_s = sorted(valid, key=lambda p: -p["score"])
    rank_e = {p["pocket"]: i for i, p in enumerate(by_e)}
    rank_s = {p["pocket"]: i for i, p in enumerate(by_s)}

    def combined(p):
        return rank_e[p["pocket"]] + LAMBDA * rank_s[p["pocket"]]

    best = min(valid, key=combined)

    box_out = Path(out_dir) / f"p2rank_rf_{case}.box"
    c = best["center"]
    with open(box_out, "w") as f:
        f.write(f"center_x = {c[0]:.3f}\ncenter_y = {c[1]:.3f}\ncenter_z = {c[2]:.3f}\n")

    info = {
        "pocket": best["pocket"], "p2rank_score": best["score"], "vina_energy": results[best["pocket"]],
        "rank_e": rank_e[best["pocket"]], "rank_s": rank_s[best["pocket"]], "combined": combined(best),
    }
    return case, best["center"], info