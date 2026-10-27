#!/usr/bin/env python3
"""
seed_rankfusion.py — 对 AF3 的 5 个 seed 做 Rank Fusion

核心思路（与 model selection 的区别）：
  - model selection：只看 ranking_score 选最高
  - Rank Fusion：融合多个信号（ranking_score + consistency + clash + buriedness）
                 给 5 个 seed 排序

候选：每个 case 的 5 个 seed
信号：
  1. ranking_score    (AF3 置信度，越高越好)
  2. consistency      (邻近 seed 数，越多越好)
  3. clash_score      (配体-蛋白最近距离，越大越好)
  4. buriedness       (配体埋藏深度，越大越好)
  5. receptor_contacts(5 Å 内受体原子数，越多越好)

融合：combined = rank_R + λ1·rank_C + λ2·rank_clash + λ3·rank_buried + λ4·rank_contacts

评估：融合后的 seed 选择 vs top1(ranking_score) vs best(oracle)

输出：
  seed_rf_results/seed_summary.csv
  seed_rf_results/seed_aggregate.csv
  seed_rf_results/seed_report.txt
"""
import argparse
import csv
import difflib
from pathlib import Path
from itertools import combinations

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


def split_cif(cif_path, out_rec, out_lig):
    st = load_structure(cif_path)
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


def get_all_atoms(pdb):
    """返回 [(pos, elem, res_key), ...]"""
    st = load_structure(pdb)
    out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element.name in ("H", "D"):
                    continue
                out.append({
                    "pos": np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                    "elem": atom.element.name.upper(),
                    "res": (chain.name, res.seqid.num, res.name),
                })
    return out


def compute_signals(lig_pdb, rec_pdb, lig_centroid):
    """
    对单个 seed 计算信号：
      - clash_score: 配体到最近受体原子的距离（越大越好）
      - buriedness: 配体质心到受体表面最近距离（越大越好）
      - receptor_contacts: 5 Å 内受体原子数（越多越好）
    """
    lig_atoms = get_all_atoms(lig_pdb)
    rec_atoms = get_all_atoms(rec_pdb)
    if not lig_atoms or not rec_atoms:
        return None

    rec_pos = np.array([a["pos"] for a in rec_atoms])

    # ---- clash score：配体所有原子到受体所有原子的最小距离 ----
    min_dist = 1e9
    for a in lig_atoms:
        d = np.linalg.norm(rec_pos - a["pos"], axis=1)
        m = float(d.min())
        if m < min_dist:
            min_dist = m

    # ---- receptor_contacts：5 Å 内受体原子数 ----
    n_contacts = 0
    for a in lig_atoms:
        d = np.linalg.norm(rec_pos - a["pos"], axis=1)
        n_contacts += int((d < 5.0).sum())

    # ---- buriedness：配体质心到受体原子的距离分布 ----
    d_center = np.linalg.norm(rec_pos - lig_centroid, axis=1)
    # 用最近 5 个原子的平均距离作为"埋藏深度"的代理
    nearest5 = np.sort(d_center)[:5]
    buriedness = float(nearest5.mean())

    return {
        "clash_score": round(min_dist, 3),
        "receptor_contacts": int(n_contacts),
        "buriedness": round(buriedness, 3),
    }


def true_center(lig_path):
    p = Path(lig_path); suf = p.suffix.lower()
    coords = []
    if suf in (".sdf", ".mol"):
        with open(p, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        n = int(lines[3][0:3].strip())
        for i in range(4, 4 + n):
            if i >= len(lines):
                break
            coords.append([float(lines[i][0:10]), float(lines[i][10:20]),
                           float(lines[i][20:30])])
    elif suf == ".pdb":
        with open(p) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    elem = line[76:78].strip().upper() if len(line) > 76 else ""
                    if elem in ("H", "D"):
                        continue
                    try:
                        coords.append([float(line[30:38]), float(line[38:46]),
                                       float(line[46:54])])
                    except ValueError:
                        continue
    return np.mean(coords, axis=0) if coords else None


def read_ranking_scores(case_dir):
    csvs = list(Path(case_dir).glob("*_ranking_scores.csv"))
    if not csvs:
        return {}
    out = {}
    try:
        with open(csvs[0], encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    s = int(r["sample"])
                    out[s] = float(r["ranking_score"])
                except (ValueError, KeyError):
                    continue
    except OSError:
        return {}
    return out


def ligand_centroid_pdb(pdb_path):
    atoms = get_all_atoms(pdb_path)
    if not atoms:
        return None
    return np.mean([a["pos"] for a in atoms], axis=0)


# =====================================================================
# 排序融合核心
# =====================================================================
def rank_fuse(seeds, weights):
    """
    seeds: list of dict，每个含 ranking_score, consistency, clash_score,
           buriedness, receptor_contacts
    weights: {"ranking": 1.0, "consistency": 0.5, "clash": 0.5,
              "buried": 0.5, "contacts": 0.5}
    返回：选中 seed 的 index
    """
    n = len(seeds)
    if n == 0:
        return -1

    def rank_of(values, descending=True):
        """返回每个 index 的排名（0 = 最好）"""
        order = sorted(range(n),
                       key=lambda i: values[i],
                       reverse=descending)
        ranks = [0] * n
        for r, i in enumerate(order):
            ranks[i] = r
        return ranks

    # 各信号的排名
    r_ranking = rank_of([s["ranking_score"] for s in seeds], descending=True)
    r_consist = rank_of([s["consistency"] for s in seeds], descending=True)
    r_clash   = rank_of([s["clash_score"] for s in seeds], descending=True)
    r_buried  = rank_of([-s["buriedness"] for s in seeds], descending=True)  # buriedness 小 = 好
    r_cont    = rank_of([s["receptor_contacts"] for s in seeds], descending=True)

    combined = [
        weights["ranking"] * r_ranking[i] +
        weights["consistency"] * r_consist[i] +
        weights["clash"] * r_clash[i] +
        weights["buried"] * r_buried[i] +
        weights["contacts"] * r_cont[i]
        for i in range(n)
    ]
    return int(np.argmin(combined))


# =====================================================================
# 单 case
# =====================================================================
def process_case(case, af3_dir, exp_dir, ligand_dir, out_dir, weights,
                 consistency_radius=3.0):
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

    tc = true_center(lig_file)
    if tc is None:
        return None, []

    ranking_scores = read_ranking_scores(af3_case)

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

        ca_raw = get_ca_list(raw_rec)
        ca_exp = get_ca_list(exp_rec)
        pairs = match_seq(ca_raw, ca_exp)
        if len(pairs) < 10:
            continue

        P = np.array([ca_raw[i_][3] for i_, _ in pairs])
        Q = np.array([ca_exp[j][3] for _, j in pairs])
        R, t = kabsch(P, Q)

        # 配体质心对齐后
        lig_c_raw = ligand_centroid_pdb(raw_lig)
        if lig_c_raw is None:
            continue
        lig_c_aligned = R @ lig_c_raw + t
        err = float(np.linalg.norm(lig_c_aligned - tc))

        # 变换受体到实验坐标系（用于算 clash 等信号）
        rec_aligned = work / f"seed{i}_rec_aligned.pdb"
        st_r = gemmi.read_structure(str(raw_rec))
        for chain in st_r[0]:
            for res in chain:
                for atom in res:
                    p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    q = R @ p + t
                    atom.pos = gemmi.Position(float(q[0]), float(q[1]),
                                              float(q[2]))
        st_r.write_pdb(str(rec_aligned))

        lig_aligned = work / f"seed{i}_lig_aligned.pdb"
        st_l = gemmi.read_structure(str(raw_lig))
        for chain in st_l[0]:
            for res in chain:
                for atom in res:
                    p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    q = R @ p + t
                    atom.pos = gemmi.Position(float(q[0]), float(q[1]),
                                              float(q[2]))
        st_l.write_pdb(str(lig_aligned))

        # 计算信号
        signals = compute_signals(str(lig_aligned), str(rec_aligned),
                                  lig_c_aligned)
        if signals is None:
            continue

        seed_rows.append({
            "case": case,
            "seed": i,
            "err_A": round(err, 3),
            "ranking_score": ranking_scores.get(i, 0.0),
            "clash_score": signals["clash_score"],
            "receptor_contacts": signals["receptor_contacts"],
            "buriedness": signals["buriedness"],
            "_lig_center": lig_c_aligned,
        })

    if len(seed_rows) < 2:
        return None, []

    # ---- consistency：以 radius 内的邻近 seed 数计算 ----
    centers = [r["_lig_center"] for r in seed_rows]
    n = len(centers)
    for i in range(n):
        cnt = 0
        for j in range(n):
            if i == j:
                continue
            d = np.linalg.norm(centers[i] - centers[j])
            if d < consistency_radius:
                cnt += 1
        seed_rows[i]["consistency"] = cnt

    # ---- 融合选择 ----
    chosen = rank_fuse(seed_rows, weights)
    top1_idx = int(np.argmax([r["ranking_score"] for r in seed_rows]))
    best_idx = int(np.argmin([r["err_A"] for r in seed_rows]))

    agg = {
        "case": case,
        "n_seeds": len(seed_rows),
        "top1_err": seed_rows[top1_idx]["err_A"],
        "best_err": seed_rows[best_idx]["err_A"],
        "rf_err": seed_rows[chosen]["err_A"],
        "top1_seed": top1_idx,
        "best_seed": best_idx,
        "rf_seed": chosen,
        "rf_is_best": 1 if chosen == best_idx else 0,
        "rf_eq_top1": 1 if chosen == top1_idx else 0,
        "mean_err": round(float(np.mean([r["err_A"] for r in seed_rows])), 3),
    }

    # 清理
    for r in seed_rows:
        r.pop("_lig_center", None)

    return agg, seed_rows


# =====================================================================
# 主流程
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--af3_dir", default="AF3_workflow")
    p.add_argument("--exp_dir", default="exp_structures")
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--out_dir", default="seed_rf_results")
    p.add_argument("--cases", default=None)
    p.add_argument("--consistency_radius", type=float, default=3.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 默认权重（ranking_score 为主，其余辅助）
    weights = {
        "ranking": 1.0,
        "consistency": 0.5,
        "clash": 0.5,
        "buried": 0.5,
        "contacts": 0.5,
    }

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        cases = sorted([d.name for d in Path(args.af3_dir).iterdir()
                        if d.is_dir()])

    print(f"共 {len(cases)} 个 case")
    print(f"输出: {out_dir.resolve()}")
    print(f"一致性半径: {args.consistency_radius} Å")
    print(f"权重: {weights}\n")

    all_agg = []
    all_seed = []
    for i, case in enumerate(cases, 1):
        if i % 20 == 0 or i == 1:
            print(f"  [{i}/{len(cases)}] {case}")
        try:
            agg, seeds = process_case(case, args.af3_dir, args.exp_dir,
                                      args.ligand_dir, args.out_dir,
                                      weights, args.consistency_radius)
        except Exception as e:
            print(f"    ⚠ {case}: {str(e)[:60]}")
            agg, seeds = None, []
        if agg:
            all_agg.append(agg)
        all_seed.extend(seeds)

    if all_agg:
        agg_csv = out_dir / "seed_aggregate.csv"
        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_agg[0].keys()))
            w.writeheader(); w.writerows(all_agg)
        print(f"\n✅ {agg_csv}  ({len(all_agg)} cases)")

    if all_seed:
        seed_csv = out_dir / "seed_summary.csv"
        with open(seed_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_seed[0].keys()))
            w.writeheader(); w.writerows(all_seed)
        print(f"✅ {seed_csv}  ({len(all_seed)} seed records)")

    if not all_agg:
        return

    # ---- 报告 ----
    L = []
    L.append("=" * 72)
    L.append("AF3 5-seed Rank Fusion 报告")
    L.append("=" * 72)
    L.append(f"\n  总 case: {len(all_agg)}")
    L.append(f"  融合权重: {weights}")

    top1 = np.array([r["top1_err"] for r in all_agg])
    best = np.array([r["best_err"] for r in all_agg])
    rf   = np.array([r["rf_err"] for r in all_agg])

    L.append("\n" + "-" * 72)
    L.append("【三种选择策略的成功率】")
    L.append("-" * 72)
    L.append(f"\n  {'策略':<25}{'均值':>10}{'中位数':>10}"
             f"{'< 2 Å':>10}{'< 4 Å':>10}")
    L.append("  " + "-" * 64)
    for name, arr in [("Top1 (ranking_score)", top1),
                      ("Best (oracle over seeds)", best),
                      ("Rank Fusion (ours)", rf)]:
        L.append(f"  {name:<25}{arr.mean():>10.2f}"
                 f"{np.median(arr):>10.2f}"
                 f"{(arr < 2).mean()*100:>9.1f}%"
                 f"{(arr < 4).mean()*100:>9.1f}%")

    L.append("\n" + "-" * 72)
    L.append("【Rank Fusion vs Top1 / Best】")
    L.append("-" * 72)

    rf_eq_top1 = sum(r["rf_eq_top1"] for r in all_agg)
    rf_is_best = sum(r["rf_is_best"] for r in all_agg)
    L.append(f"\n  RF 与 Top1 选同一个 seed:  {rf_eq_top1}/{len(all_agg)} "
             f"({rf_eq_top1/len(all_agg)*100:.1f}%)")
    L.append(f"  RF 选中真正的 best seed:   {rf_is_best}/{len(all_agg)} "
             f"({rf_is_best/len(all_agg)*100:.1f}%)")

    # 成功率对比
    rf_rate = (rf < 4).mean() * 100
    top1_rate = (top1 < 4).mean() * 100
    best_rate = (best < 4).mean() * 100

    L.append("\n" + "=" * 72)
    L.append("【关键结论】")
    L.append("=" * 72)
    L.append(f"\n  Top1 成功率:               {top1_rate:.2f}%")
    L.append(f"  Rank Fusion 成功率:        {rf_rate:.2f}%")
    L.append(f"  Best (oracle over seeds):  {best_rate:.2f}%")
    L.append(f"\n  RF - Top1 = {rf_rate - top1_rate:+.2f} 个百分点")
    L.append(f"  RF - Best = {rf_rate - best_rate:+.2f} 个百分点")

    if rf_rate > top1_rate + 2:
        L.append(f"\n  ★ Rank Fusion 显著优于 Top1：融合多信号能选出更好的 seed")
    elif abs(rf_rate - top1_rate) <= 2:
        L.append(f"\n   Rank Fusion 与 Top1 相当：5 个 seed 高度一致，融合无额外增益")
    else:
        L.append(f"\n   Rank Fusion 略差于 Top1：融合信号噪声过大")

    if best_rate > rf_rate + 5:
        L.append(f"\n   但 Best 明显高于 RF：5 个 seed 中有更好答案，融合未充分利用")
        L.append(f"   → 可探索更好的融合权重")

    text = "\n".join(L)
    print("\n" + text)
    report_path = out_dir / "seed_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n✅ {report_path}")


if __name__ == "__main__":
    main()