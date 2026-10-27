#!/usr/bin/env python3
"""
probe_pocket.py — 探测 AF3 结构真值口袋位置的几何空腔

原理：
  1. 把真值配体质心从实验坐标 反变换 到 AF3 原始坐标
  2. 以该点为中心，计算能容纳的最大探针球半径
     probe_radius = min_i(d_i - vdw_i)   # i 遍历周围所有重原子
  3. 与实验结构同一位置对比
  4. 找出堵住口袋的残基
  5. 与 fpocket 找到的最近口袋对比

输出：
  probe_results/probe_summary.csv
  probe_results/probe_report.txt
"""
import argparse
import csv
import difflib
from pathlib import Path

import numpy as np
import gemmi

AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
       "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}

# 范德华半径（Å）
VDW = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85,
    "I": 1.98, "H": 1.20,
}


def vdw_of(elem):
    return VDW.get(elem.upper(), 1.70)


def load_structure(path):
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()
    return st


def split_cif(cif, out_rec):
    st = load_structure(cif)
    prot = []
    for chain in st[0]:
        is_prot = any(r.name.strip().upper() in AA3 for r in chain)
        if is_prot:
            prot.append(chain.name)
    st_p = st.clone()
    for chain in st[0]:
        if chain.name not in prot:
            st_p[0].remove_chain(chain.name)
    st_p.write_pdb(str(out_rec))


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


def all_atoms_with_vdw(pdb_path, radius, center):
    """返回 center 半径 radius 内所有重原子的 (pos, vdw, res_key)"""
    st = load_structure(pdb_path)
    c = np.array(center, dtype=float)
    out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element.name in ("H", "D"):
                    continue
                p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                if np.linalg.norm(p - c) < radius:
                    out.append({
                        "pos": p,
                        "vdw": vdw_of(atom.element.name),
                        "res": (chain.name, res.seqid.num, res.name),
                        "atom": atom.name,
                    })
    return out


def max_probe_radius(center, atoms):
    """在 center 处能容纳的最大球半径"""
    c = np.array(center, dtype=float)
    max_r = 5.0
    for a in atoms:
        d = float(np.linalg.norm(a["pos"] - c))
        r = d - a["vdw"]
        if r < max_r:
            max_r = r
    return max(max_r, 0.0)


def find_blocking_atoms(center, atoms, n=5):
    """找出离中心最近的几个原子及其残基"""
    c = np.array(center, dtype=float)
    scored = []
    for a in atoms:
        d = float(np.linalg.norm(a["pos"] - c))
        clearance = d - a["vdw"]
        scored.append((clearance, d, a["res"], a["atom"]))
    scored.sort()
    return scored[:n]


def read_box_center(box_path):
    with open(box_path) as f:
        txt = f.read()
    vals = {}
    for line in txt.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                vals[k.strip().lower()] = float(v.strip())
            except ValueError:
                pass
    if all(k in vals for k in ("center_x", "center_y", "center_z")):
        return np.array([vals["center_x"], vals["center_y"], vals["center_z"]])
    return None


# =====================================================================
# 主流程
# =====================================================================
def process_case(case, af3_dir, exp_dir, ligand_dir, fpocket_dir, out_dir):
    af3_case = Path(af3_dir) / case
    cifs = sorted(af3_case.rglob("*.cif")) if af3_case.is_dir() else []
    if not cifs:
        return {"case": case, "status": "no_cif"}
    cif = cifs[0]

    exp_rec = Path(exp_dir) / case / "receptor.pdb"
    if not exp_rec.is_file():
        return {"case": case, "status": "no_exp_rec"}

    lig_case = Path(ligand_dir) / case
    lig_file = None
    for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb"):
        p = lig_case / nm
        if p.is_file():
            lig_file = p
            break
    if lig_file is None:
        return {"case": case, "status": "no_lig"}

    tc = true_center(lig_file)
    if tc is None:
        return {"case": case, "status": "no_true_center"}

    work = Path(out_dir) / case
    work.mkdir(parents=True, exist_ok=True)
    raw_rec = work / "af3_rec_raw.pdb"

    try:
        split_cif(cif, raw_rec)
    except Exception as e:
        return {"case": case, "status": f"split_fail_{str(e)[:30]}"}

    # Kabsch：raw → exp
    ca_raw = get_ca_list(raw_rec)
    ca_exp = get_ca_list(exp_rec)
    pairs = match_seq(ca_raw, ca_exp)
    if len(pairs) < 10:
        return {"case": case, "status": f"few_ca_{len(pairs)}"}

    P = np.array([ca_raw[i][3] for i, _ in pairs])
    Q = np.array([ca_exp[j][3] for _, j in pairs])
    R, t = kabsch(P, Q)

    # 实验坐标系的真值口袋中心 → AF3 坐标系
    R_inv = R.T
    t_inv = -R_inv @ t
    tc_af3 = R_inv @ tc + t_inv

    # 探测 AF3 结构
    atoms_af3 = all_atoms_with_vdw(raw_rec, radius=10.0, center=tc_af3)
    probe_af3 = max_probe_radius(tc_af3, atoms_af3)
    blockers_af3 = find_blocking_atoms(tc_af3, atoms_af3, n=5)

    # 探测实验结构同一位置
    atoms_exp = all_atoms_with_vdw(exp_rec, radius=10.0, center=tc)
    probe_exp = max_probe_radius(tc, atoms_exp)

    # fpocket 在 AF3 结构上找到的口袋
    fpocket_center = None
    fpocket_dist = None
    if fpocket_dir:
        ranks_json = Path(fpocket_dir) / case / "af3_top1" / "pocket_ranks.json"
        if ranks_json.is_file():
            import json
            with open(ranks_json) as f:
                pockets = json.load(f)
            # 找离真值最近的 fpocket 口袋
            best_d = 1e9
            for p in pockets:
                c_af3 = np.array(p["center"])
                d = float(np.linalg.norm(c_af3 - tc_af3))
                if d < best_d:
                    best_d = d
                    fpocket_center = c_af3
            fpocket_dist = best_d

    # 记录
    return {
        "case": case,
        "status": "ok",
        "probe_af3_A": round(probe_af3, 2),
        "probe_exp_A": round(probe_exp, 2),
        "probe_delta": round(probe_af3 - probe_exp, 2),
        "n_atoms_af3": len(atoms_af3),
        "n_atoms_exp": len(atoms_exp),
        "top_blocker": f"{blockers_af3[0][2][2]}{blockers_af3[0][2][1]}:{blockers_af3[0][3]}"
            if blockers_af3 else "",
        "top_clearance": round(blockers_af3[0][0], 2) if blockers_af3 else "",
        "fpocket_min_dist": round(fpocket_dist, 2) if fpocket_dist else "",
    }


# =====================================================================
# 主入口
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--af3_dir", default="AF3_workflow")
    p.add_argument("--exp_dir", default="exp_structures")
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--fpocket_dir", default="results",
                   help="fpocket + Rank Fusion 结果目录（含 pocket_ranks.json）")
    p.add_argument("--out_dir", default="probe_results")
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

    rows = []
    for i, case in enumerate(cases, 1):
        if i % 20 == 0 or i == 1:
            print(f"  [{i}/{len(cases)}] {case}")
        try:
            r = process_case(case, args.af3_dir, args.exp_dir,
                             args.ligand_dir, args.fpocket_dir, args.out_dir)
        except Exception as e:
            r = {"case": case, "status": f"crash_{str(e)[:40]}"}
        rows.append(r)

    csv_path = out_dir / "probe_summary.csv"
    keys = ["case", "status", "probe_af3_A", "probe_exp_A", "probe_delta",
            "n_atoms_af3", "n_atoms_exp", "top_blocker", "top_clearance",
            "fpocket_min_dist"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n✅ {csv_path}")

    # =========================================================
    # 报告
    # =========================================================
    L = []
    L.append("=" * 72)
    L.append("真值口袋位置的几何空腔探测报告")
    L.append("=" * 72)

    ok = [r for r in rows if r.get("status") == "ok"]
    L.append(f"\n  总 case: {len(rows)}")
    L.append(f"  成功:    {len(ok)}")
    L.append(f"  失败:    {len(rows) - len(ok)}")

    if not ok:
        print("\n".join(L))
        return

    # ---- 探针半径分布 ----
    pa = np.array([r["probe_af3_A"] for r in ok])
    pe = np.array([r["probe_exp_A"] for r in ok])
    pd = np.array([r["probe_delta"] for r in ok])

    L.append("\n" + "-" * 72)
    L.append("【探针半径分布】")
    L.append("-" * 72)
    L.append(f"\n  真值口袋位置的最大探针球半径（Å）:")
    L.append(f"    AF3 结构:   mean={pa.mean():.2f}  median={np.median(pa):.2f}  "
             f"Q1={np.percentile(pa,25):.2f}  Q3={np.percentile(pa,75):.2f}")
    L.append(f"    实验结构:   mean={pe.mean():.2f}  median={np.median(pe):.2f}  "
             f"Q1={np.percentile(pe,25):.2f}  Q3={np.percentile(pe,75):.2f}")
    L.append(f"    差值(AF3-exp): mean={pd.mean():+.2f}  median={np.median(pd):+.2f}")

    # ---- 分档 ----
    L.append("\n" + "-" * 72)
    L.append("【AF3 探针半径分档】")
    L.append("-" * 72)
    bins = [(0, 1), (1, 2), (2, 3), (3, 5)]
    L.append(f"\n  {'半径 (Å)':<15}{'n':>6}{'占比':>10}{'含义'}")
    L.append("  " + "-" * 60)
    for lo, hi in bins:
        n = int(((pa >= lo) & (pa < hi)).sum())
        pct = n / len(pa) * 100
        meaning = {
            (0, 1): "空腔完全被堵",
            (1, 2): "水分子都进不去",
            (2, 3): "勉强能容纳小分子",
            (3, 5): "空腔足够",
        }[(lo, hi)]
        L.append(f"  [{lo}, {hi})          {n:>6}{pct:>9.1f}%   {meaning}")

    # ---- 对比 ----
    L.append("\n" + "-" * 72)
    L.append("【AF3 vs 实验】")
    L.append("-" * 72)
    n_af3_narrow = int((pd < -0.5).sum())
    n_af3_wide = int((pd > 0.5).sum())
    n_same = len(pd) - n_af3_narrow - n_af3_wide
    L.append(f"\n  AF3 空腔更窄 (Δ < -0.5 Å):  {n_af3_narrow} "
             f"({n_af3_narrow/len(pd)*100:.1f}%)")
    L.append(f"  AF3 空腔更宽 (Δ > +0.5 Å):  {n_af3_wide} "
             f"({n_af3_wide/len(pd)*100:.1f}%)")
    L.append(f"  相近 (|Δ| ≤ 0.5 Å):         {n_same} "
             f"({n_same/len(pd)*100:.1f}%)")

    # ---- fpocket 最小距离 ----
    fdist = [r["fpocket_min_dist"] for r in ok if r.get("fpocket_min_dist") != ""]
    if fdist:
        fdist = np.array(fdist)
        L.append("\n" + "-" * 72)
        L.append("【fpocket 找到的最近口袋到真值中心的距离】")
        L.append("-" * 72)
        L.append(f"\n  n = {len(fdist)}")
        L.append(f"  mean   = {fdist.mean():.2f} Å")
        L.append(f"  median = {np.median(fdist):.2f} Å")
        L.append(f"  < 4 Å:  {(fdist < 4).sum()} ({(fdist < 4).mean()*100:.1f}%)")
        L.append(f"  < 8 Å:  {(fdist < 8).sum()} ({(fdist < 8).mean()*100:.1f}%)")

    # ---- 堵住口袋的残基 TOP 10 ----
    L.append("\n" + "-" * 72)
    L.append("【最常堵住真值口袋的残基】")
    L.append("-" * 72)
    from collections import Counter
    blocker_counts = Counter(r["top_blocker"] for r in ok
                             if r.get("top_blocker"))
    L.append(f"\n  {'残基':<20}{'出现次数':>10}")
    L.append("  " + "-" * 35)
    for k, v in blocker_counts.most_common(10):
        L.append(f"  {k:<20}{v:>10}")

    # ---- 结论 ----
    L.append("\n" + "=" * 72)
    L.append("【结论】")
    L.append("=" * 72)
    if np.median(pa) < 2.0:
        L.append(f"\n  AF3 结构在真值口袋位置的最大探针半径中位数为 "
                 f"{np.median(pa):.2f} Å，小于水分子半径（1.4 Å）+ 余量。")
        L.append(f"  说明 AF3 结构的侧链位置偏差使真值口袋被堵住，")
        L.append(f"  fpocket 基于几何空腔的检测策略因此失效。")
        L.append(f"\n  → 结论：AF3 结构上 fpocket 失效的原因是"
                 f"「口袋几何被改变」，")
        L.append(f"    而非对齐或排序问题。")
    else:
        L.append(f"\n  AF3 结构在真值口袋位置仍有足够空腔"
                 f"（中位数 {np.median(pa):.2f} Å），")
        L.append(f"  fpocket 检测失败可能是打分或排序问题，需要进一步分析。")

    text = "\n".join(L)
    print("\n" + text)
    rp = out_dir / "probe_report.txt"
    with open(rp, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n✅ {rp}")


if __name__ == "__main__":
    main()