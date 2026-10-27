#!/usr/bin/env python3
"""
p2rank_dump.py — 跑一次 P2Rank 前 20 pocket 的 Vina 快速对接，
保存所有 pocket 级数据到 CSV。

外层串行遍历 case，内层并行跑每个 pocket（避免 Windows 上嵌套 pool 死锁）。
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "pipeline"))

import af3_common as ac
import auto_pipeline as ap


P2RANK_OUT_ROOT = Path(r"E:\AF3环境\p2rank_work\p2rank_out")
WORK_ROOT = Path(r"E:\AF3环境\p2rank_dump_work")

MAX_POCKETS = 20
BOX_SIZE = 25.0
QUICK_EXH = 4
QUICK_MODES = 3
SEED = 42
INNER_WORKERS = 8   # 内层并行数，按 CPU 核数调整


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
                    "pocket": row["name"],
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


def _dock_one(args):
    """在子进程里执行单个 pocket 的 Vina 对接。"""
    import subprocess
    rec_pdbqt, lig_pdbqt, center, out_path, pocket_name = args
    cmd = [
        ap.VINA_BIN, "--receptor", rec_pdbqt, "--ligand", lig_pdbqt,
        "--out", out_path,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", str(BOX_SIZE), "--size_y", str(BOX_SIZE),
        "--size_z", str(BOX_SIZE),
        "--exhaustiveness", str(QUICK_EXH),
        "--num_modes", str(QUICK_MODES), "--seed", str(SEED),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return pocket_name, None
    return pocket_name, ap.read_best_energy(out_path)


def process_one(case, exp_dir, ligand_dir):
    """处理单个 case：准备受体/配体 + 并行跑 20 个 pocket。"""
    exp_rec = Path(exp_dir) / case / "receptor.pdb"
    lig_case = Path(ligand_dir) / case
    if not exp_rec.is_file() or not lig_case.is_dir():
        return []

    lig_file = None
    for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb", "ligand.mol2"):
        pth = lig_case / nm
        if pth.is_file():
            lig_file = pth
            break
    if lig_file is None:
        return []

    true_c = ac.ligand_centroid(lig_file)
    if true_c is None:
        return []

    pockets = load_top_pockets(case, P2RANK_OUT_ROOT)
    if not pockets:
        return []

    work = WORK_ROOT / case
    work.mkdir(parents=True, exist_ok=True)

    rec_pdbqt = work / "receptor.pdbqt"
    if not rec_pdbqt.is_file() or rec_pdbqt.stat().st_size < 100:
        if not ap.pdb_to_pdbqt_receptor(str(exp_rec), str(rec_pdbqt)):
            return []

    ligs = sorted(lig_case.glob("*.pdb"))
    if not ligs:
        return []

    lig_pdbqts = []
    for lig in ligs:
        out_pdbqt = work / f"{lig.stem}.pdbqt"
        if not out_pdbqt.is_file() or out_pdbqt.stat().st_size < 100:
            if not ap.pdb_to_pdbqt_ligand(str(lig), str(out_pdbqt)):
                continue
        lig_pdbqts.append(out_pdbqt)
    if not lig_pdbqts:
        return []

    def count_atoms(p):
        try:
            return sum(1 for l in open(p) if l.startswith(("ATOM", "HETATM")))
        except OSError:
            return 0
    lig_pdbqts.sort(key=count_atoms)
    probe = lig_pdbqts[0]

    tasks = [(str(rec_pdbqt), str(probe), p["center"],
              str(work / f"quick_{p['pocket']}.pdbqt"), p["pocket"])
             for p in pockets]

    results = {}
    with ProcessPoolExecutor(max_workers=INNER_WORKERS) as ex:
        futures = {ex.submit(_dock_one, t): t[4] for t in tasks}
        for fut in as_completed(futures):
            name, e = fut.result()
            results[name] = e

    rows = []
    for p in pockets:
        e = results.get(p["pocket"])
        err = float(np.linalg.norm(p["center"] - true_c))
        rows.append({
            "case": case, "pocket": p["pocket"], "rank": p["rank"],
            "p2rank_score": round(p["score"], 4),
            "vina_energy": round(e, 3) if e is not None else "",
            "center_x": round(p["center"][0], 3),
            "center_y": round(p["center"][1], 3),
            "center_z": round(p["center"][2], 3),
            "err_to_true": round(err, 3),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", default="exp_structures")
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--cases", default=None)
    p.add_argument("--resume", action="store_true",
                   help="断点续跑：跳过已处理的 case")
    args = p.parse_args()

    exp_dir = Path(args.exp_dir)
    ligand_dir = Path(args.ligand_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        cases = sorted([d.name for d in ligand_dir.iterdir() if d.is_dir()])
    print(f"共 {len(cases)} 个 case")

    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    # 断点续跑：从已有 CSV 里读出已完成的 case
    done_cases = set()
    csv_path = out_dir / "p2rank_pocket_data.csv"
    if args.resume and csv_path.is_file():
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done_cases.add(r["case"])
        print(f"已完成: {len(done_cases)} 个，跳过")
        cases = [c for c in cases if c not in done_cases]

    # 追加模式：如果续跑，先读旧数据
    all_rows = []
    if args.resume and csv_path.is_file():
        with open(csv_path, encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))

    # ---- 外层串行，内层并行 ----
    for i, case in enumerate(cases, 1):
        try:
            rows = process_one(case, exp_dir, ligand_dir)
            all_rows.extend(rows)
            print(f"  [{i}/{len(cases)}] {case}: {len(rows)} pockets")
        except Exception as e:
            print(f"  [{i}/{len(cases)}] {case}: ERROR {str(e)[:60]}")

        # 每 20 个 case 写一次 CSV（防止中断丢数据）
        if i % 20 == 0 or i == len(cases):
            keys = ["case", "pocket", "rank", "p2rank_score", "vina_energy",
                    "center_x", "center_y", "center_z", "err_to_true"]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(all_rows)
            print(f"    [saved] {csv_path}  ({len(all_rows)} rows)")

    print(f"\n[OK] {csv_path}  共 {len(all_rows)} 行")


if __name__ == "__main__":
    main()