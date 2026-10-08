#!/usr/bin/env python3
"""
run_p2rank_rankfusion.py — P2Rank + RankFusion

流程：
  1. 从 P2Rank 输出读取每个 case 的前 N 个 pocket（默认 20）
  2. 每个 pocket center 跑 Vina 快速对接（exh=4）
  3. RankFusion 融合 P2Rank score 排序与 Vina 能量排序（λ=0.75）
  4. 输出 results/p2rank_rf_{case}.box

输出：
  results/p2rank_rf_{case}.box
  results/p2rank_rf_summary.csv
"""
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

import auto_pipeline as ap          # noqa: E402


# ============================================================
# 配置
# ============================================================
P2RANK_OUT_ROOT = Path(r"E:\AF3环境\p2rank_work\p2rank_out")
WORK_ROOT = Path(r"E:\AF3环境\p2rank_rf_work")

MAX_POCKETS = 20
LAMBDA = 0.75
BOX_SIZE = 25.0
QUICK_EXH = 4
QUICK_MODES = 3
SEED = 42


# ============================================================
# 1. 读取 P2Rank 输出
# ============================================================
def load_p2rank_pockets(case, p2rank_root):
    """返回该 case 的前 MAX_POCKETS 个 pocket。

    每个 pocket: {"pocket": str, "rank": int, "score": float, "center": np.array}
    """
    root = Path(p2rank_root)
    # P2Rank 每个 case 输出在 {case}/{case}.pdb_predictions.csv
    candidates = list(root.rglob(f"{case}.pdb_predictions.csv"))
    if not candidates:
        return []
    pred_csv = candidates[0]

    pockets = []
    with open(pred_csv, encoding="utf-8") as f:
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
    return pockets[:MAX_POCKETS]


# ============================================================
# 2. 单口袋 Vina 快速对接
# ============================================================
def _dock_one(args):
    """(rec_pdbqt, lig_pdbqt, center, out_path, pocket_name) → (pocket_name, energy)"""
    rec_pdbqt, lig_pdbqt, center, out_path, pocket_name = args
    cmd = [
        ap.VINA_BIN,
        "--receptor", rec_pdbqt,
        "--ligand", lig_pdbqt,
        "--out", out_path,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", str(BOX_SIZE),
        "--size_y", str(BOX_SIZE),
        "--size_z", str(BOX_SIZE),
        "--exhaustiveness", str(QUICK_EXH),
        "--num_modes", str(QUICK_MODES),
        "--seed", str(SEED),
    ]
    import subprocess
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return pocket_name, None
    return pocket_name, ap.read_best_energy(out_path)


# ============================================================
# 3. 单 case 处理
# ============================================================
def process_one(case, exp_dir, ligand_dir, out_dir):
    """返回 (case, best_center_or_None, info)"""
    exp_rec = Path(exp_dir) / case / "receptor.pdb"
    lig_case = Path(ligand_dir) / case
    if not exp_rec.is_file() or not lig_case.is_dir():
        return case, None, "missing input"

    pockets = load_p2rank_pockets(case, P2RANK_OUT_ROOT)
    if not pockets:
        return case, None, "no p2rank output"

    # 工作目录
    work = WORK_ROOT / case
    work.mkdir(parents=True, exist_ok=True)

    # 受体准备
    rec_pdbqt = work / "receptor.pdbqt"
    if not rec_pdbqt.is_file() or rec_pdbqt.stat().st_size < 100:
        if not ap.pdb_to_pdbqt_receptor(str(exp_rec), str(rec_pdbqt)):
            return case, None, "receptor pdbqt failed"

    # 配体准备：用最小的配体做 probe
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

    # 并行快速对接
    tasks = []
    for p in pockets:
        out_pdbqt = work / f"quick_{p['pocket']}.pdbqt"
        tasks.append((str(rec_pdbqt), str(probe),
                      p["center"], str(out_pdbqt), p["pocket"]))

    results = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_dock_one, t): t[4] for t in tasks}
        for fut in as_completed(futures):
            name, e = fut.result()
            results[name] = e

    # RankFusion
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

    # 写 box
    box_out = Path(out_dir) / f"p2rank_rf_{case}.box"
    c = best["center"]
    with open(box_out, "w") as f:
        f.write(f"center_x = {c[0]:.3f}\n")
        f.write(f"center_y = {c[1]:.3f}\n")
        f.write(f"center_z = {c[2]:.3f}\n")

    info = {
        "pocket": best["pocket"],
        "p2rank_score": best["score"],
        "vina_energy": results[best["pocket"]],
        "rank_e": rank_e[best["pocket"]],
        "rank_s": rank_s[best["pocket"]],
        "combined": combined(best),
    }
    return case, best["center"], info


# ============================================================
# 4. 主流程
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", default="exp_structures")
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--cases", default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        # 从 P2Rank 输出目录里挑出所有 case
        if not P2RANK_OUT_ROOT.is_dir():
            print(f"⚠ P2Rank 输出目录不存在: {P2RANK_OUT_ROOT}")
            return
        cases = sorted([d.name for d in P2RANK_OUT_ROOT.iterdir() if d.is_dir()])

    # 断点续跑：已有 box 就跳过
    cases = [c for c in cases
             if not (out_dir / f"p2rank_rf_{c}.box").is_file()]
    print(f"待处理: {len(cases)} 个 case")

    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    n_ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, c, exp_dir, args.ligand_dir, out_dir): c
                   for c in cases}
        done = 0
        for fut in as_completed(futures):
            case, center, info = fut.result()
            done += 1
            if center is not None:
                n_ok += 1
                rows.append({"case": case,
                             "p2rank_rf_score": round(info["p2rank_score"], 4),
                             "vina_energy": round(info["vina_energy"], 3),
                             "box_written": 1})
            else:
                rows.append({"case": case, "p2rank_rf_score": "",
                             "vina_energy": "", "box_written": 0})
                print(f"  [FAIL] {case}: {info}")
            if done % 20 == 0 or done == len(cases):
                print(f"  进度 {done}/{len(cases)}  成功 {n_ok}")

    csv_path = out_dir / "p2rank_rf_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "p2rank_rf_score",
                                          "vina_energy", "box_written"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n✅ 成功 {n_ok}/{len(rows)}")
    print(f"✅ {csv_path}")


if __name__ == "__main__":
    main()