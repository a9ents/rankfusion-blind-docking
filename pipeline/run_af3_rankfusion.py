
import argparse
import csv
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# 路径补丁：让 pipeline/ 和 src/ 都能被找到
# ---------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # .../pipeline
_ROOT = _THIS_DIR.parent                             # .../ALL-DATA
sys.path.insert(0, str(_ROOT / "src"))               # 找 af3_common
sys.path.insert(0, str(_THIS_DIR))                   # 找 auto_pipeline

import af3_common as ac            # noqa: E402  (来自 src/)
import auto_pipeline as ap         # noqa: E402  (来自 pipeline/)


def find_af3_cifs(af3_root, case_name):
    case_dir = Path(af3_root) / case_name
    if not case_dir.is_dir():
        return []
    return sorted(case_dir.rglob("*.cif"))


def score_and_align_models(cifs, exp_receptor, work_dir):
    """对每个 CIF 打分 + 叠合，返回候选列表。"""
    candidates = []
    for i, cif in enumerate(cifs):
        conf = ac.load_af3_summary_confidence(cif)
        ranking = conf.get("ranking_score",
                           conf.get("overall_confidence", -1.0))
        iptm = conf.get("iptm", -1.0)

        raw_pdb = work_dir / f"_model_{i}_raw.pdb"
        lig_pdb = work_dir / f"_model_{i}_lig.pdb"
        ok_r, ok_l = ac.split_af3_cif(cif, raw_pdb, lig_pdb)
        if not ok_r:
            continue

        aligned = work_dir / f"_model_{i}_aligned.pdb"
        rmsd, tm = ac.run_usalign(raw_pdb, exp_receptor, aligned)

        candidates.append({
            "index": i,
            "cif": cif,
            "aligned_pdb": aligned if aligned.is_file() else None,
            "ligand_pdb": lig_pdb if ok_l else None,
            "rmsd": rmsd if rmsd is not None else float("inf"),
            "tm": tm,
            "ranking_score": float(ranking) if ranking is not None else -1.0,
            "iptm": float(iptm) if iptm is not None else -1.0,
        })
    return candidates


def run_rankfusion(receptor_pdb, ligand_dir, work_dir, lam=0.75):
    """pocket_guided（fpocket + Rank Fusion + Vina）。"""
    ap.CFG["blind_mode"] = "pocket_guided"
    ap.CFG["pg_rank_lambda"] = lam
    ap.CFG["pg_quick_exh"] = 4
    ap.CFG["pg_quick_modes"] = 3
    ap.CFG["exhaustiveness_fine"] = 32
    ap.CFG["num_modes_fine"] = 20
    ok, _ = ap.run_pocket_guided(str(receptor_pdb), str(ligand_dir), str(work_dir))
    return ok


def process_one(case_name, af3_root, exp_dir, ligand_dir, out_dir):
    print(f"\n{'=' * 70}\n[{case_name}]\n{'=' * 70}")

    exp_receptor = Path(exp_dir) / case_name / "receptor.pdb"
    lig_case_dir = Path(ligand_dir) / case_name
    if not exp_receptor.is_file():
        print(f"  ✗ 缺少实验受体: {exp_receptor}")
        return []
    if not lig_case_dir.is_dir():
        print(f"  ✗ 缺少配体目录: {lig_case_dir}")
        return []

    work = Path(out_dir) / case_name
    work.mkdir(parents=True, exist_ok=True)

    cifs = find_af3_cifs(af3_root, case_name)
    if not cifs:
        print("  ✗ 未找到 AF3 模型")
        return []

    print(f"  找到 {len(cifs)} 个 AF3 模型")
    cands = score_and_align_models(cifs, exp_receptor, work)
    if not cands:
        print("  ✗ 全部叠合失败")
        return []

    # 主报：ranking_score 最高；上界：RMSD 最小
    top1 = max(cands, key=lambda c: c["ranking_score"])
    oracle = min(cands, key=lambda c: c["rmsd"])

    print(f"  AF3-top1   : {top1['cif'].name}  "
          f"rank={top1['ranking_score']:.3f}  RMSD={top1['rmsd']:.2f}")
    print(f"  RMSD-oracle: {oracle['cif'].name}  "
          f"rank={oracle['ranking_score']:.3f}  RMSD={oracle['rmsd']:.2f}")

    # --- AF3 top1 → Rank Fusion ---
    if top1["aligned_pdb"]:
        sub = work / "af3_top1"
        sub.mkdir(exist_ok=True)
        rec = sub / "receptor.pdb"
        shutil.copy(top1["aligned_pdb"], rec)
        if run_rankfusion(rec, lig_case_dir, sub):
            box = sub / "receptor.box"
            if box.is_file():
                shutil.copy(box, Path(out_dir) / f"af3top1_rankfusion_{case_name}.box")
                print(f"  ✅ af3top1_rankfusion_{case_name}.box")

    # --- RMSD-oracle → Rank Fusion（上界）---
    if oracle["aligned_pdb"] and oracle["index"] != top1["index"]:
        sub = work / "af3_oracle"
        sub.mkdir(exist_ok=True)
        rec = sub / "receptor.pdb"
        shutil.copy(oracle["aligned_pdb"], rec)
        if run_rankfusion(rec, lig_case_dir, sub):
            box = sub / "receptor.box"
            if box.is_file():
                shutil.copy(box, Path(out_dir) / f"af3oracle_rankfusion_{case_name}.box")
                print(f"  ✅ af3oracle_rankfusion_{case_name}.box")

    # --- AF3 共折叠配体 → 盒子（AF3-only baseline）---
    if top1["ligand_pdb"] and top1["aligned_pdb"]:
        dst_pdb = Path(out_dir) / f"af3ligand_{case_name}.pdb"
        try:
            _, _ = ac.run_usalign(top1["ligand_pdb"], top1["aligned_pdb"],
                                  dst_pdb)
        except Exception as e:
            print(f"  ⚠ af3ligand 叠合失败: {e}")
            shutil.copy(top1["ligand_pdb"], dst_pdb)
        c = ac.ligand_centroid(dst_pdb)
        if c is not None:
            with open(Path(out_dir) / f"af3ligand_{case_name}.box", "w") as f:
                f.write(f"center_x = {c[0]:.3f}\n")
                f.write(f"center_y = {c[1]:.3f}\n")
                f.write(f"center_z = {c[2]:.3f}\n")
            print(f"  ✅ af3ligand_{case_name}.box")

    # --- 汇总行 ---
    rows = []
    for tag, c in (("af3_top1", top1), ("af3_rmsd_oracle", oracle)):
        rows.append({
            "case": case_name,
            "model": tag,
            "cif": c["cif"].name,
            "ranking_score": round(c["ranking_score"], 4),
            "iptm": round(c["iptm"], 4),
            "rmsd_to_exp": round(c["rmsd"], 3) if c["rmsd"] != float("inf") else "",
            "tm_score": round(c["tm"], 4) if c["tm"] is not None else "",
            "has_ligand": c["ligand_pdb"] is not None,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--af3_output", default="AF3_workflow")
    p.add_argument("--exp_dir", default="exp_structures")
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--cases", default=None)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        root = Path(args.af3_output)
        cases = sorted([d.name for d in root.iterdir() if d.is_dir()])

    print(f"发现 {len(cases)} 个复合物")
    all_rows = []
    for i, name in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {name}")
        all_rows.extend(
            process_one(name, args.af3_output, args.exp_dir,
                        args.ligand_dir, args.out_dir)
        )

    if all_rows:
        csv_path = Path(args.out_dir) / "af3_model_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n✅ {csv_path}")


if __name__ == "__main__":
    main()