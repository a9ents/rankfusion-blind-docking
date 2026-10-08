
"""
probe_deviation.py — 量化 AF3 配体位置偏离

核心问题：
  1. 偏离多少？（质心距离 + 全原子 RMSD）
  2. 往哪偏？（5 个 seed 是否系统性地往同一方向偏）
  3. 偏离和 AF3 置信度的关系

输出：
  deviation_results/per_seed.csv       每个 seed 详细指标
  deviation_results/per_case.csv       每个 case 汇总
  deviation_results/report.txt         报告
  deviation_results/figs/              图
"""
import argparse
import csv
import difflib
from pathlib import Path

import numpy as np
import gemmi

AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
       "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}


def load_structure(path):
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()
    return st


def split_cif(cif, out_rec, out_lig):
    st = load_structure(cif)
    prot, lig = [], []
    for chain in st[0]:
        is_prot = any(r.name.strip().upper() in AA3 for r in chain)
        (prot if is_prot else lig).append(chain.name)
    if not prot or not lig:
        return False, False
    st_p = st.clone()
    for ch in lig:
        st_p[0].remove_chain(ch)
    st_p.write_pdb(str(out_rec))
    st_l = st.clone()
    for ch in prot:
        st_l[0].remove_chain(ch)
    st_l.write_pdb(str(out_lig))
    return True, True


def get_ca_list(pdb):
    st = load_structure(pdb)
    out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.name == "CA":
                    out.append((chain.name, res.seqid.num, res.name,
                                np.array([atom.pos.x, atom.pos.y, atom.pos.z])))
                    break
    return out


def match_seq(a, b):
    sa = [x[2] for x in a]
    sb = [x[2] for x in b]
    sm = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
    pairs = []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            pairs.append((blk.a + k, blk.b + k))
    return pairs


def kabsch(P, Q):
    Pc = P.mean(0); Qc = Q.mean(0)
    P0 = P - Pc; Q0 = Q - Qc
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc


def get_atoms(pdb, exclude_h=True):
    st = load_structure(pdb)
    out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if exclude_h and atom.element.name in ("H", "D"):
                    continue
                out.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
    return np.array(out) if out else None


def read_atoms_from_sdf(sdf_path):
    p = Path(sdf_path)
    coords = []
    with open(p, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    n = int(lines[3][0:3].strip())
    for i in range(4, 4 + n):
        if i >= len(lines):
            break
        line = lines[i]
        if len(line) < 30:
            continue
        coords.append([float(line[0:10]), float(line[10:20]),
                       float(line[20:30])])
    return np.array(coords) if coords else None


def read_atoms_from_pdb(pdb_path):
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                elem = line[76:78].strip().upper() if len(line) > 76 else ""
                if elem in ("H", "D"):
                    continue
                try:
                    coords.append([float(line[30:38]),
                                   float(line[38:46]),
                                   float(line[46:54])])
                except ValueError:
                    continue
    return np.array(coords) if coords else None


def read_true_ligand(lig_file):
    suf = Path(lig_file).suffix.lower()
    if suf in (".sdf", ".mol"):
        return read_atoms_from_sdf(lig_file)
    if suf in (".pdb", ".ent"):
        return read_atoms_from_pdb(lig_file)
    return None


def read_ranking_score(case_dir, sample_idx):
    csvs = list(Path(case_dir).glob("*_ranking_scores.csv"))
    if not csvs:
        return None
    try:
        with open(csvs[0], encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if int(r["sample"]) == sample_idx:
                    return float(r["ranking_score"])
    except Exception:
        return None
    return None


def read_iptm(cif_path):
    import json
    d = Path(cif_path).parent
    cands = []
    for name in ("summary_confidences.json", "summary_confidences_0.json"):
        p = d / name
        if p.is_file():
            cands.append(p)
    for p in d.glob("*SUMMARY_CONFIDENCES*.json"):
        cands.append(p)
    for p in cands:
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f).get("iptm")
        except Exception:
            continue
    return None


# =====================================================================
# 主流程
# =====================================================================
def process_case(case, af3_dir, exp_dir, ligand_dir, out_dir):
    af3_case = Path(af3_dir) / case
    if not af3_case.is_dir():
        return None, []

    exp_rec = Path(exp_dir) / case / "receptor.pdb"
    if not exp_rec.is_file():
        return None, []

    lig_case = Path(ligand_dir) / case
    lig_file = None
    for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb"):
        p = lig_case / nm
        if p.is_file():
            lig_file = p
            break
    if lig_file is None:
        return None, []

    # 真值配体
    true_atoms = read_true_ligand(str(lig_file))
    if true_atoms is None:
        return None, []
    true_centroid = true_atoms.mean(axis=0)

    # 实验受体原子（用于计算"到表面距离"）
    exp_atoms = get_atoms(exp_rec)

    work = Path(out_dir) / case
    work.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    for i in range(5):
        sample_dir = af3_case / f"SEED-1_SAMPLE-{i}"
        if not sample_dir.is_dir():
            continue
        cifs = sorted(sample_dir.glob("*.cif"))
        if not cifs:
            continue
        cif = cifs[0]

        raw_rec = work / f"seed{i}_rec.pdb"
        raw_lig = work / f"seed{i}_lig.pdb"
        try:
            ok_r, ok_l = split_cif(cif, raw_rec, raw_lig)
        except Exception:
            continue
        if not ok_r or not ok_l:
            continue

        # 受体 Cα 匹配
        ca_raw = get_ca_list(raw_rec)
        ca_exp = get_ca_list(exp_rec)
        pairs = match_seq(ca_raw, ca_exp)
        if len(pairs) < 10:
            continue

        P = np.array([ca_raw[k][3] for k, _ in pairs])
        Q = np.array([ca_exp[j][3] for _, j in pairs])
        R, t = kabsch(P, Q)

        # AF3 配体原子
        af3_atoms = get_atoms(raw_lig)
        if af3_atoms is None or len(af3_atoms) == 0:
            continue
        af3_atoms_aligned = (R @ af3_atoms.T).T + t
        af3_centroid = af3_atoms_aligned.mean(axis=0)

        # 1. 质心距离
        centroid_dist = float(np.linalg.norm(af3_centroid - true_centroid))

        # 2. 偏离方向（单位向量）
        diff = af3_centroid - true_centroid
        norm = np.linalg.norm(diff)
        if norm > 1e-6:
            direction = diff / norm
        else:
            direction = np.zeros(3)

        # 3. 全原子 RMSD（原子数相同时才计算）
        ligand_rmsd = None
        if len(af3_atoms_aligned) == len(true_atoms):
            d = af3_atoms_aligned - true_atoms
            ligand_rmsd = float(np.sqrt((d ** 2).sum(axis=1).mean()))

        # 4. 到受体表面最近距离（AF3 配体质心）
        min_dist_to_surface = None
        if exp_atoms is not None and len(exp_atoms) > 0:
            dists = np.linalg.norm(exp_atoms - af3_centroid, axis=1)
            min_dist_to_surface = float(dists.min())

        # 5. 真值配体质心到表面的距离（对照）
        true_min_dist = None
        if exp_atoms is not None and len(exp_atoms) > 0:
            dists = np.linalg.norm(exp_atoms - true_centroid, axis=1)
            true_min_dist = float(dists.min())

        ranking = read_ranking_score(af3_case, i)
        iptm = read_iptm(cif)

        seed_rows.append({
            "case": case,
            "seed": i,
            "centroid_dist": round(centroid_dist, 3),
            "ligand_rmsd": round(ligand_rmsd, 3) if ligand_rmsd is not None else "",
            "dir_x": round(float(direction[0]), 4),
            "dir_y": round(float(direction[1]), 4),
            "dir_z": round(float(direction[2]), 4),
            "min_dist_to_surface": round(min_dist_to_surface, 3)
                if min_dist_to_surface is not None else "",
            "true_min_dist_to_surface": round(true_min_dist, 3)
                if true_min_dist is not None else "",
            "ranking_score": round(ranking, 4) if ranking is not None else "",
            "iptm": round(iptm, 4) if iptm is not None else "",
        })

    if len(seed_rows) < 2:
        return None, []

    # ---- 方向一致性分析 ----
    dirs = np.array([[r["dir_x"], r["dir_y"], r["dir_z"]] for r in seed_rows])
    mean_dir = dirs.mean(axis=0)
    mean_dir_norm = np.linalg.norm(mean_dir)
    if mean_dir_norm > 1e-6:
        mean_dir = mean_dir / mean_dir_norm

    angles = []
    for d in dirs:
        dot = float(np.clip(np.dot(d, mean_dir), -1.0, 1.0))
        angles.append(np.degrees(np.arccos(dot)))

    # 所有 seed 到质心距离
    dists = [r["centroid_dist"] for r in seed_rows]
    rmsds = [r["ligand_rmsd"] for r in seed_rows if r["ligand_rmsd"] != ""]

    # 平均方向角
    mean_angle = float(np.mean(angles))

    # 方向一致性（平均 1 - cos(angle)）
    consistency_metric = float(np.mean([1 - np.cos(np.radians(a)) for a in angles]))

    agg = {
        "case": case,
        "n_seeds": len(seed_rows),
        "centroid_dist_mean": round(float(np.mean(dists)), 3),
        "centroid_dist_median": round(float(np.median(dists)), 3),
        "centroid_dist_min": round(float(min(dists)), 3),
        "centroid_dist_max": round(float(max(dists)), 3),
        "centroid_dist_std": round(float(np.std(dists)), 3),
        "ligand_rmsd_mean": round(float(np.mean(rmsds)), 3) if rmsds else "",
        "ligand_rmsd_median": round(float(np.median(rmsds)), 3) if rmsds else "",
        "mean_angle_to_consensus": round(mean_angle, 2),
        "directional_consistency": round(consistency_metric, 4),
        "min_dist_to_surface_mean": round(float(np.mean(
            [r["min_dist_to_surface"] for r in seed_rows
             if r["min_dist_to_surface"] != ""])), 3)
            if any(r["min_dist_to_surface"] != "" for r in seed_rows) else "",
        "true_min_dist_to_surface": seed_rows[0]["true_min_dist_to_surface"],
        "iptm": seed_rows[0]["iptm"],
    }
    return agg, seed_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--af3_dir", default="AF3_workflow")
    p.add_argument("--exp_dir", default="exp_structures")
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--out_dir", default="deviation_results")
    p.add_argument("--cases", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        cases = sorted([d.name for d in Path(args.af3_dir).iterdir()
                        if d.is_dir()])

    print(f"共 {len(cases)} 个 case")
    print(f"输出: {out_dir.resolve()}\n")

    all_agg = []
    all_seed = []
    for i, case in enumerate(cases, 1):
        if i % 20 == 0 or i == 1:
            print(f"  [{i}/{len(cases)}] {case}")
        try:
            agg, seeds = process_case(case, args.af3_dir, args.exp_dir,
                                      args.ligand_dir, args.out_dir)
        except Exception as e:
            print(f"    ⚠ {case}: {str(e)[:60]}")
            agg, seeds = None, []
        if agg:
            all_agg.append(agg)
        all_seed.extend(seeds)

    if not all_agg:
        print("⚠ 无结果")
        return

    # 写 CSV
    agg_csv = out_dir / "per_case.csv"
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_agg[0].keys()))
        w.writeheader(); w.writerows(all_agg)
    print(f"\n✅ {agg_csv}  ({len(all_agg)} cases)")

    seed_csv = out_dir / "per_seed.csv"
    with open(seed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_seed[0].keys()))
        w.writeheader(); w.writerows(all_seed)
    print(f"✅ {seed_csv}  ({len(all_seed)} seeds)")

    # ---- 报告 ----
    L = []
    L.append("=" * 72)
    L.append("AF3 配体位置偏离分析")
    L.append("=" * 72)

    cd = np.array([r["centroid_dist_mean"] for r in all_agg])
    rmsd = np.array([r["ligand_rmsd_mean"] for r in all_agg
                     if r["ligand_rmsd_mean"] != ""])
    angles = np.array([r["mean_angle_to_consensus"] for r in all_agg])
    consistency = np.array([r["directional_consistency"] for r in all_agg])

    L.append("\n" + "-" * 72)
    L.append("【质心距离分布】")
    L.append("-" * 72)
    L.append(f"\n  n = {len(cd)}")
    L.append(f"  mean   = {cd.mean():.2f} Å")
    L.append(f"  median = {np.median(cd):.2f} Å")
    L.append(f"  Q1–Q3  = {np.percentile(cd, 25):.2f} – "
             f"{np.percentile(cd, 75):.2f} Å")

    bins = [(0, 2), (2, 4), (4, 6), (6, 10), (10, 20), (20, 100)]
    L.append(f"\n  {'区间 (Å)':<14}{'n':>6}{'占比':>10}")
    L.append("  " + "-" * 30)
    for lo, hi in bins:
        n = int(((cd >= lo) & (cd < hi)).sum())
        L.append(f"  [{lo}, {hi}){'':<6}{n:>6}{n/len(cd)*100:>9.1f}%")

    if len(rmsd) > 0:
        L.append("\n" + "-" * 72)
        L.append("【全原子 RMSD 分布（Kabsch 对齐后）】")
        L.append("-" * 72)
        L.append(f"\n  n = {len(rmsd)}")
        L.append(f"  mean   = {rmsd.mean():.2f} Å")
        L.append(f"  median = {np.median(rmsd):.2f} Å")
        L.append(f"  Q1–Q3  = {np.percentile(rmsd, 25):.2f} – "
                 f"{np.percentile(rmsd, 75):.2f} Å")

    L.append("\n" + "-" * 72)
    L.append("【方向一致性分析（关键）】")
    L.append("-" * 72)
    L.append(f"\n  mean_angle_to_consensus（5 个 seed 偏离方向的平均夹角）:")
    L.append(f"    mean   = {angles.mean():.1f}°")
    L.append(f"    median = {np.median(angles):.1f}°")
    L.append(f"    Q1–Q3  = {np.percentile(angles, 25):.1f}° – "
             f"{np.percentile(angles, 75):.1f}°")

    # 判断系统 vs 随机
    n_systematic = int((angles < 30).sum())
    n_random = int((angles > 60).sum())
    n_intermediate = len(angles) - n_systematic - n_random

    L.append(f"\n  系统性偏离（夹角 < 30°）: {n_systematic} "
             f"({n_systematic/len(angles)*100:.1f}%)")
    L.append(f"  中间状态（30–60°）:        {n_intermediate} "
             f"({n_intermediate/len(angles)*100:.1f}%)")
    L.append(f"  随机偏离（夹角 > 60°）:   {n_random} "
             f"({n_random/len(angles)*100:.1f}%)")

    if n_systematic / len(angles) > 0.5:
        L.append(f"\n  → 结论：多数 case 的 5 个 seed 系统性往同一方向偏。")
        L.append(f"    这证明 AF3 的偏差是系统性的，不是随机采样噪声。")
    elif n_random / len(angles) > 0.5:
        L.append(f"\n  → 结论：多数 case 的 5 个 seed 随机往不同方向偏。")
        L.append(f"    说明偏离来自采样波动，不是系统性漂移。")
    else:
        L.append(f"\n  → 结论：系统性偏离与随机偏离混合，无统一模式。")

    # 表面距离分析
    surf = [r["min_dist_to_surface_mean"] for r in all_agg
            if r["min_dist_to_surface_mean"] != ""]
    true_surf = [r["true_min_dist_to_surface"] for r in all_agg
                 if r["true_min_dist_to_surface"] != ""]
    if surf and true_surf:
        L.append("\n" + "-" * 72)
        L.append("【配体质心到受体表面最近距离】")
        L.append("-" * 72)
        surf_arr = np.array(surf)
        true_surf_arr = np.array(true_surf)
        L.append(f"\n  AF3 配体:  mean = {surf_arr.mean():.2f} Å, "
                 f"median = {np.median(surf_arr):.2f} Å")
        L.append(f"  真值配体: mean = {true_surf_arr.mean():.2f} Å, "
                 f"median = {np.median(true_surf_arr):.2f} Å")
        L.append(f"  差值: {np.median(surf_arr) - np.median(true_surf_arr):+.2f} Å")
        if np.median(surf_arr) > np.median(true_surf_arr) + 2:
            L.append(f"\n  → AF3 配体普遍浮在蛋白表面外侧（> 真值 2 Å）")
        elif np.median(surf_arr) < np.median(true_surf_arr) - 2:
            L.append(f"\n  → AF3 配体普遍埋在蛋白内部过深（< 真值 2 Å）")
        else:
            L.append(f"\n  → AF3 配体埋藏深度与真值相近")

    text = "\n".join(L)
    print("\n" + text)
    rp = out_dir / "report.txt"
    with open(rp, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n✅ {rp}")


if __name__ == "__main__":
    main()