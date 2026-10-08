import argparse
import csv
import difflib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import gemmi

# ================= 从 af3_common 读 USALIGN_EXE =================
try:
    import af3_common
    USALIGN_EXE = af3_common.USALIGN_EXE
except ImportError:
    USALIGN_EXE = r"E:\AF3环境\USalign\USalign.exe"

print(f"[info] USALIGN_EXE = {USALIGN_EXE}")
if not Path(USALIGN_EXE).is_file():
    print(f"[ERROR] USalign 不存在: {USALIGN_EXE}")
    raise SystemExit(1)


# =====================================================================
# 拆 CIF
# =====================================================================
def split_receptor_ligand(cif_path, out_receptor, out_ligand):
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()

    aa3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
           "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
    nt3 = {"A","C","G","U","DA","DC","DG","DT"}

    prot_chains, lig_chains = [], []
    for chain in st[0]:
        is_prot = any(
            (r.name.strip().upper() in aa3) or (r.name.strip().upper() in nt3)
            for r in chain
        )
        (prot_chains if is_prot else lig_chains).append(chain.name)

    if not prot_chains:
        return False, False

    st_prot = st.clone()
    for ch in lig_chains:
        st_prot[0].remove_chain(ch)
    st_prot.write_pdb(str(out_receptor))

    ligand_ok = False
    if lig_chains:
        st_lig = st.clone()
        for ch in prot_chains:
            st_lig[0].remove_chain(ch)
        st_lig.write_pdb(str(out_ligand))
        ligand_ok = True
    return True, ligand_ok


# =====================================================================
# USalign
# =====================================================================
def run_usalign(mobile, target, out_pdb, timeout=120):
    prefix = str(Path(out_pdb).with_suffix(""))
    cmd = [USALIGN_EXE, str(Path(mobile).resolve()),
           str(Path(target).resolve()), "-mm", "1", "-o", prefix]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8",
                           errors="replace")
    except FileNotFoundError:
        print(f"    USALIGN not found: {USALIGN_EXE}")
        return None
    except subprocess.TimeoutExpired:
        print("    USALIGN timeout")
        return None

    if r.returncode != 0:
        print(f"    USALIGN rc={r.returncode}")
        if r.stderr:
            print(f"    stderr: {r.stderr[:300]}")
        return None

    actual = Path(prefix + ".pdb")
    if actual.is_file() and actual != Path(out_pdb):
        shutil.move(str(actual), str(out_pdb))
    for f in actual.parent.glob(actual.stem + "*.pml"):
        try:
            f.unlink()
        except OSError:
            pass
    return out_pdb if Path(out_pdb).is_file() else None


# =====================================================================
# Cα 提取 + 序列匹配
# =====================================================================
def get_ca_list(pdb_path):
    """返回有序 [(chain, seqid, resname, coords), ...]"""
    st = gemmi.read_structure(str(pdb_path))
    out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.name == "CA":
                    out.append((chain.name, res.seqid.num, res.name,
                                [atom.pos.x, atom.pos.y, atom.pos.z]))
                    break
    return out


def match_ca(raw_pdb, aligned_pdb):
    """按残基序列顺序匹配 Cα（不依赖编号）。"""
    kr = get_ca_list(raw_pdb)
    ka = get_ca_list(aligned_pdb)
    if not kr or not ka:
        return None, None

    seq_r = [x[2] for x in kr]
    seq_a = [x[2] for x in ka]

    sm = difflib.SequenceMatcher(None, seq_r, seq_a, autojunk=False)
    P, Q = [], []
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            P.append(kr[block.a + k][3])
            Q.append(ka[block.b + k][3])

    if len(P) < 3:
        return None, None
    return np.array(P), np.array(Q)


def get_all_coords(pdb_path):
    st = gemmi.read_structure(str(pdb_path))
    coords = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element.name in ("H", "D"):
                    continue
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.array(coords) if coords else None


# =====================================================================
# Kabsch
# =====================================================================
def kabsch(P, Q):
    """求 R, t 使得 Q ≈ R @ P + t"""
    Pc = P.mean(axis=0); Qc = Q.mean(axis=0)
    P0 = P - Pc; Q0 = Q - Qc
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def transform_pdb(in_pdb, R, t, out_pdb):
    st = gemmi.read_structure(str(in_pdb))
    for chain in st[0]:
        for res in chain:
            for atom in res:
                p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                q = R @ p + t
                atom.pos = gemmi.Position(float(q[0]), float(q[1]), float(q[2]))
    st.write_pdb(str(out_pdb))


# =====================================================================
# 盒子 / 质心
# =====================================================================
def write_box(center, size, path):
    with open(path, "w") as f:
        f.write(f"center_x = {center[0]:.3f}\n")
        f.write(f"center_y = {center[1]:.3f}\n")
        f.write(f"center_z = {center[2]:.3f}\n")
        f.write(f"size_x = {size:.1f}\n")
        f.write(f"size_y = {size:.1f}\n")
        f.write(f"size_z = {size:.1f}\n")


def ligand_centroid_simple(pdb_path):
    c = get_all_coords(pdb_path)
    return c.mean(axis=0) if c is not None else None


def sdf_centroid(sdf_path):
    try:
        with open(sdf_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return None
    if len(lines) < 4:
        return None

    # 用 -1 作为哨兵值，避免 None 造成的类型推断问题
    n_atoms: int = -1
    atom_start: int = -1

    try:
        n_atoms = int(lines[3][0:3].strip())
        atom_start = 4
    except (ValueError, IndexError):
        for i, line in enumerate(lines[:10]):
            if len(line) >= 3 and line[0:3].strip().isdigit():
                n_atoms = int(line[0:3].strip())
                atom_start = i + 1
                break

    if n_atoms <= 0:
        return None
    if atom_start < 0:
        return None

    coords = []
    for i in range(atom_start, atom_start + n_atoms):
        if i >= len(lines):
            break
        line = lines[i]
        if len(line) < 30:
            continue
        try:
            x = float(line[0:10].strip())
            y = float(line[10:20].strip())
            z = float(line[20:30].strip())
            coords.append([x, y, z])
        except (ValueError, IndexError):
            continue

    return np.mean(coords, axis=0) if coords else None


def pdb_centroid(pdb_path):
    coords = []
    with open(pdb_path) as f:
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


def ligand_centroid_file(path):
    p = Path(path); suf = p.suffix.lower()
    if suf in (".sdf", ".mol"):
        return sdf_centroid(p)
    if suf in (".pdb", ".ent"):
        return pdb_centroid(p)
    return None


# =====================================================================
# 单 case
# =====================================================================
def process_case(case, af3_dir, exp_dir, ligand_dir, out_dir, box_size=25.0):
    af3_case = Path(af3_dir) / case
    cifs = sorted(af3_case.rglob("*.cif")) if af3_case.is_dir() else []
    if not cifs:
        return None
    cif = cifs[0]

    exp_rec = Path(exp_dir) / case / "receptor.pdb"
    if not exp_rec.is_file():
        return None

    lig_case = Path(ligand_dir) / case
    lig_file = None
    for name in ("ligand.sdf", "ligand.mol", "ligand.pdb", "ligand.mol2"):
        p = lig_case / name
        if p.is_file():
            lig_file = p
            break
    if lig_file is None:
        return None

    work = Path(out_dir) / case / "fixlig"
    work.mkdir(parents=True, exist_ok=True)

    raw_rec = work / "af3_rec_raw.pdb"
    raw_lig = work / "af3_lig_raw.pdb"
    ok_r, ok_l = split_receptor_ligand(cif, raw_rec, raw_lig)
    if not ok_r or not ok_l:
        return None

    aligned_rec = work / "af3_rec_aligned.pdb"
    if run_usalign(raw_rec, exp_rec, aligned_rec) is None:
        return None

    # 关键：用序列匹配，不靠编号
    P, Q = match_ca(raw_rec, aligned_rec)
    if P is None:
        return None
    R, t = kabsch(P, Q)

    aligned_lig = Path(out_dir) / f"af3ligand_fixed_{case}.pdb"
    transform_pdb(raw_lig, R, t, aligned_lig)

    c = ligand_centroid_simple(aligned_lig)
    if c is None:
        return None
    box_path = Path(out_dir) / f"af3ligand_fixed_{case}.box"
    write_box(c, box_size, box_path)

    true_c = ligand_centroid_file(lig_file)
    err = float(np.linalg.norm(c - true_c)) if true_c is not None else None

    return {"case": case,
            "err_A": round(err, 3) if err is not None else "",
            "lig_file": str(lig_file),
            "true_c_ok": true_c is not None,
            "n_ca": len(P)}


# =====================================================================
# 主流程
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--af3_dir", default="AF3_workflow")
    p.add_argument("--exp_dir", default="exp_structures")
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--out_dir", default="fixligand_results")
    p.add_argument("--cases", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        cases = sorted([d.name for d in Path(args.af3_dir).iterdir()
                        if d.is_dir()])

    print(f"共 {len(cases)} 个 case")
    print(f"输出目录: {out_dir.resolve()}\n")

    rows = []
    for i, case in enumerate(cases, 1):
        if i % 20 == 0 or i == 1:
            print(f"  [{i}/{len(cases)}] {case}")
        try:
            r = process_case(case, args.af3_dir, args.exp_dir,
                             args.ligand_dir, args.out_dir)
        except Exception as e:
            print(f"    ⚠ {case}: {e}")
            r = None
        if r:
            rows.append(r)

    if not rows:
        print("\n⚠ 没有任何 case 成功")
        return

    csv_path = out_dir / "af3ligand_fixed_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n✅ {csv_path}")

    # 诊断
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.get("true_c_ok"))
    print(f"\n=== 诊断 ===")
    print(f"  总成功 case: {n_total}")
    print(f"  真值中心可算: {n_ok}")
    print(f"  真值中心不可算: {n_total - n_ok}")

    errs = [float(r["err_A"]) for r in rows if r["err_A"] != ""]
    if errs:
        arr = np.array(errs)
        print(f"\nAF3 共折叠配体位置误差（n={len(arr)}）：")
        print(f"  mean   = {arr.mean():.2f} Å")
        print(f"  median = {np.median(arr):.2f} Å")
        print(f"  < 4 Å: {(arr < 4).sum()}  ({(arr < 4).mean() * 100:.2f}%)")
        print(f"  < 2 Å: {(arr < 2).sum()}  ({(arr < 2).mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()