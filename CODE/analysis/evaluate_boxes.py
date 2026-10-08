import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# 路径补丁：让 src/ 里的 af3_common 可以被找到
_THIS_DIR = Path(__file__).resolve().parent          # .../analysis
_ROOT = _THIS_DIR.parent                             # .../ALL-DATA
sys.path.insert(0, str(_ROOT / "src"))

import af3_common as ac                              # noqa: E402

METHODS = [
    "exp_top1",
    "exp_energy",
    "exp_rankfusion",
    "p2rank_top1",
    "p2rank_energy",         # ← 新增
    "p2rank_rf",
    "af3top1_rankfusion",
    "af3oracle_rankfusion",
    "af3ligand",
]


def find_ligand_file(case_dir):
    for name in ("ligand.sdf", "ligand.mol", "ligand.pdb", "ligand.mol2"):
        p = Path(case_dir) / name
        if p.is_file():
            return p
    return None


def evaluate_one(case, ligand_dir, box_dir):
    ligand_file = find_ligand_file(Path(ligand_dir) / case)
    if ligand_file is None:
        return None
    true_c = ac.ligand_centroid(ligand_file)
    if true_c is None:
        return None

    row = {"case": case}
    for m in METHODS:
        bp = Path(box_dir) / f"{m}_{case}.box"
        if bp.is_file():
            c = ac.parse_box_center(bp)
            row[f"{m}_err"] = round(ac.euclidean(c, true_c), 3) if c is not None else ""
        else:
            row[f"{m}_err"] = ""
    return row


def summarize(errors_by_method, thresholds=(2.0, 4.0)):
    out = {}
    for m, errs in errors_by_method.items():
        errs = [e for e in errs if e not in ("", None)]
        if not errs:
            continue
        arr = np.array(errs, dtype=float)
        out[m] = {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }
        for t in thresholds:
            out[m][f"lt_{int(t)}A"] = float(np.mean(arr < t))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--box_dir", default="results")
    p.add_argument("--out_csv", default="results/evaluation.csv")
    args = p.parse_args()

    cases = sorted([d.name for d in Path(args.ligand_dir).iterdir() if d.is_dir()])
    print(f"找到 {len(cases)} 个 case")

    rows = []
    for case in cases:
        r = evaluate_one(case, args.ligand_dir, args.box_dir)
        if r is not None:
            rows.append(r)

    if not rows:
        print("无可评估 case")
        return

    fieldnames = ["case"] + [f"{m}_err" for m in METHODS]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"✅ {out_csv}")

    errors_by_method = {m: [r.get(f"{m}_err", "") for r in rows] for m in METHODS}
    summary = summarize(errors_by_method)

    summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "n", "mean_A", "median_A", "std_A", "lt_2A", "lt_4A"])
        for m, s in summary.items():
            w.writerow([m, s["n"],
                        f"{s['mean']:.3f}", f"{s['median']:.3f}", f"{s['std']:.3f}",
                        f"{s.get('lt_2A', 0):.4f}", f"{s.get('lt_4A', 0):.4f}"])
    print(f"✅ {summary_csv}")

    print(f"\n{'method':<25}{'n':>5}{'mean':>10}{'median':>10}{'<2Å':>9}{'<4Å':>9}")
    for m, s in summary.items():
        print(f"{m:<25}{s['n']:>5}{s['mean']:>10.3f}{s['median']:>10.3f}"
              f"{s.get('lt_2A', 0) * 100:>8.2f}%{s.get('lt_4A', 0) * 100:>8.2f}%")


if __name__ == "__main__":
    main()