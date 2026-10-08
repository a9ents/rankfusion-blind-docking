import argparse
import csv
import difflib
import shutil
import sys
import subprocess
from pathlib import Path

import numpy as np
import gemmi

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
import auto_pipeline as ap        # noqa: E402


AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
       "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}


# =====================================================================
# 结构加载
# =====================================================================
def load_structure(path):
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()
    return st


def split_receptor_ligand(cif_path, out_rec, out_lig):
    st = load_structure(cif_path)
    prot, lig = [], []
    for chain in st[0]:
        is_prot = any(r.name.strip().upper() in AA3 for r in chain)
        (prot if is_prot else lig).append(chain.name)

    if not prot:
        return False, False

    st_p = st.clone()
    for ch in lig:
        st_p[0].remove_chain(ch)
    st_p.write_pdb(str(out_rec))

    ok_l = False
    if lig:
        st_l = st.clone()
        for ch in prot:
            st_l[0].remove_chain(ch)
        st_l.write_pdb(str(out_lig))
        ok_l = True
    return True, ok_l


# =====================================================================
# Cα 提取 + 序列匹配
# =====================================================================
def get_ca_list(pdb_path):
    st = load_structure(pdb_path)
    out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.name == "CA":
                    out.append((chain.name, res.seqid.num, res.name,
                                np.array([atom.pos.x, atom.pos.y, atom.pos.z])))
                    break
    return out


def match_by_sequence(list_a, list_b):
    seq_a = [x[2] for x in list_a]
    seq_b = [x[2] for x in list_b]
    sm = difflib.SequenceMatcher(None, seq_a, seq_b, autojunk=False)
    pairs = []
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            pairs.append((block.a + k, block.b + k))
    return pairs


# =====================================================================
# Kabsch
# =====================================================================
def kabsch(P, Q):
    Pc = P.mean(0); Qc = Q.mean(0)
    P0 = P - Pc; Q0 = Q - Qc
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Qc - R @ Pc
    return R, t


# =====================================================================
# 应用变换：直接改 PDB 文本坐标（保留格式）
# =====================================================================
def apply_transform_pdb_text(in_pdb, R, t, out_pdb):
    """逐行替换 PDB 坐标，保留原始格式。"""
    with open(in_pdb) as f:
        lines = f.readlines()

    out = []
    n_atoms = 0
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                p = np.array([x, y, z])
                q = R @ p + t
                if np.isnan(q).any() or np.isinf(q).any():
                    out.append(line)
                    continue
                # 只替换坐标列 (30:54)，其余原样保留
                new_line = line[:30] + f"{q[0]:8.3f}{q[1]:8.3f}{q[2]:8.3f}" + line[54:]
                out.append(new_line)
                n_atoms += 1
            except ValueError:
                out.append(line)
        else:
            out.append(line)

    # 确保 END
    if not any(l.strip() == "END" for l in out[-3:]):
        out.append("END\n")

    with open(out_pdb, "w") as f:
        f.writelines(out)
    return n_atoms


# =====================================================================
# 质心 / 口袋残基
# =====================================================================
def ligand_centroid(path):
    p = Path(path); suf = p.suffix.lower()
    if suf in (".sdf", ".mol"):
        coords = []
        try:
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
        except Exception:
            return None
        return np.mean(coords, axis=0) if coords else None
    if suf in (".pdb", ".ent"):
        coords = []
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
    return None


def pocket_residues_from_center(exp_rec_path, center, radius=8.0):
    st = load_structure(exp_rec_path)
    c = gemmi.Position(float(center[0]), float(center[1]), float(center[2]))
    keys = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.pos.dist(c) < radius:
                    keys.append((chain.name, res.seqid.num, res.name))
                    break
    return keys


def read_box_center(box_path):
    p = Path(box_path)
    if not p.is_file():
        return None
    with open(p) as f:
        txt = f.read()
    vals = {}
    for line in txt.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip().lower()
            try:
                vals[k] = float(v.strip())
            except ValueError:
                pass
    if all(k in vals for k in ("center_x", "center_y", "center_z")):
        return np.array([vals["center_x"], vals["center_y"], vals["center_z"]])
    nums = []
    for tok in txt.replace("=", " ").split():
        try:
            nums.append(float(tok))
        except ValueError:
            pass
    if len(nums) >= 3:
        return np.array(nums[:3])
    return None


# =====================================================================
# 单 case
# =====================================================================
def process_case(case, af3_dir, exp_dir, ligand_dir, out_dir, radius=8.0):
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
    for name in ("ligand.sdf", "ligand.mol", "ligand.pdb", "ligand.mol2"):
        p = lig_case / name
        if p.is_file():
            lig_file = p
            break
    if lig_file is None:
        return {"case": case, "status": "no_lig"}

    true_center = ligand_centroid(lig_file)
    if true_center is None:
        return {"case": case, "status": "no_true_center"}

    # 提取 AF3 受体
    work = Path(out_dir) / case
    work.mkdir(parents=True, exist_ok=True)

    raw_rec = work / "af3_rec_raw.pdb"
    raw_lig = work / "af3_lig_raw.pdb"
    ok_r, ok_l = split_receptor_ligand(cif, raw_rec, raw_lig)
    if not ok_r:
        return {"case": case, "status": "split_rec_fail"}
    if not ok_l:
        return {"case": case, "status": "split_lig_fail"}

    # 真值口袋残基
    pocket_keys = pocket_residues_from_center(exp_rec, true_center, radius)
    if not pocket_keys:
        return {"case": case, "status": "no_pocket_res"}

    # 序列匹配
    exp_ca = get_ca_list(exp_rec)
    af3_ca = get_ca_list(raw_rec)

    pocket_set = set(pocket_keys)
    exp_pocket_ca = [x for x in exp_ca if (x[0], x[1], x[2]) in pocket_set]

    pairs = match_by_sequence(exp_pocket_ca, af3_ca)
    if len(pairs) < 5:
        return {"case": case, "status": f"few_ca_{len(pairs)}"}

    P = np.array([exp_pocket_ca[i][3] for i, _ in pairs])
    Q = np.array([af3_ca[j][3] for _, j in pairs])

    # 检查坐标合法性
    if np.isnan(P).any() or np.isnan(Q).any():
        return {"case": case, "status": "nan_coords"}

    # Kabsch
    R, t = kabsch(Q, P)

    if np.isnan(R).any() or np.isnan(t).any():
        return {"case": case, "status": "kabsch_nan"}

    # 应用变换（文本替换）
    aligned_rec = work / "af3_rec_oracle.pdb"
    n_moved = apply_transform_pdb_text(raw_rec, R, t, aligned_rec)

    if n_moved < 50:
        return {"case": case, "status": f"few_atoms_moved_{n_moved}"}

    # 诊断：检查坐标范围
    with open(aligned_rec) as f:
        xs, ys, zs = [], [], []
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
                except ValueError:
                    pass
    if xs:
        x_range = max(xs) - min(xs)
        if x_range > 1000 or x_range < 5:
            return {"case": case, "status": f"bad_coord_range_{x_range:.0f}"}

    # 跑 Rank Fusion
    rf_work = work / "rf"
    rf_work.mkdir(parents=True, exist_ok=True)

    ap.CFG["blind_mode"] = "pocket_guided"
    ap.CFG["pg_rank_lambda"] = 0.75
    ap.CFG["pg_quick_exh"] = 4
    ap.CFG["pg_quick_modes"] = 3
    ap.CFG["exhaustiveness_fine"] = 32
    ap.CFG["num_modes_fine"] = 20

    try:
        ok, _ = ap.run_pocket_guided(str(aligned_rec), str(lig_case),
                                     str(rf_work))
    except Exception as e:
        return {"case": case, "status": f"rf_exc_{str(e)[:40]}"}

    if not ok:
        return {"case": case, "status": "rf_fail"}

    # run_pocket_guided 把 box 写到 receptor.pdb 同目录同名文件
    box_candidates = list(work.glob("*.box")) + list(rf_work.glob("*.box"))
    if not box_candidates:
        return {"case": case, "status": "no_box"}
    box_src = box_candidates[0]

    box_center = read_box_center(box_src)
    if box_center is None:
        return {"case": case, "status": "box_parse_fail"}

    err = float(np.linalg.norm(box_center - true_center))

    box_dst = Path(out_dir) / f"oracle_rf_{case}.box"
    shutil.copy(box_src, box_dst)

    return {
        "case": case,
        "status": "ok",
        "n_ca": len(pairs),
        "n_pocket_res": len(pocket_keys),
        "err_A": round(err, 3),
        "hit_4A": 1 if err < 4.0 else 0,
    }


# =====================================================================
# 主流程
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--af3_dir", default="AF3_workflow")
    p.add_argument("--exp_dir", default="exp_structures")
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--out_dir", default="oracle_results")
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--cases", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.quiet:
        ap.CFG["verbose"] = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        cases = sorted([d.name for d in Path(args.af3_dir).iterdir()
                        if d.is_dir()])

    print(f"共 {len(cases)} 个 case")
    print(f"输出: {out_dir.resolve()}")
    print(f"口袋半径: {args.radius} Å\n")

    rows = []
    for i, case in enumerate(cases, 1):
        if i % 20 == 0 or i == 1:
            print(f"  [{i}/{len(cases)}] {case}")
        try:
            r = process_case(case, args.af3_dir, args.exp_dir,
                             args.ligand_dir, args.out_dir, args.radius)
        except Exception as e:
            r = {"case": case, "status": f"crash_{str(e)[:40]}"}
        rows.append(r)

    csv_path = out_dir / "oracle_summary.csv"
    all_keys = ["case", "status", "n_ca", "n_pocket_res", "err_A", "hit_4A"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n✅ {csv_path}")

    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") != "ok"]

    print(f"\n=== 结果 ===")
    print(f"  成功: {len(ok)}/{len(rows)}")

    if fail:
        from collections import Counter
        c = Counter(r["status"] for r in fail)
        print(f"  失败原因:")
        for k, v in c.most_common(8):
            print(f"    {k}: {v}")

    if ok:
        errs = np.array([r["err_A"] for r in ok])
        hits = np.array([r["hit_4A"] for r in ok])
        print(f"\n=== Oracle 局部对齐 + Rank Fusion ===")
        print(f"  n      = {len(ok)}")
        print(f"  mean   = {errs.mean():.2f} Å")
        print(f"  median = {np.median(errs):.2f} Å")
        print(f"  < 2 Å  = {(errs < 2).sum()}  ({(errs < 2).mean()*100:.2f}%)")
        print(f"  < 4 Å  = {hits.sum()}  ({hits.mean()*100:.2f}%)")

    print(f"\n=== 对比 ===")
    print(f"  实验结构 Rank Fusion:         36.2%")
    print(f"  AF3 全局对齐 Rank Fusion:     4.97%")
    if ok:
        print(f"  AF3 oracle 局部对齐:          {hits.mean()*100:.2f}%")


if __name__ == "__main__":
    main()