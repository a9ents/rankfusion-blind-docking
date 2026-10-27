import os
import glob
import csv
import numpy as np
import gemmi
import difflib
from pathlib import Path

VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "H": 1.20}
def vdw_of(elem): return VDW.get(elem.upper(), 1.70)

# ⚠️ 请确认并修改这里的路径
CIF_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3_workflow"
APO_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD"
RELAXED_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD-DONE"
EXP_DIR = r"E:\AF3环境\ALL-DATA\DATA\exp_structures" # 实验受体文件夹
LIG_DIR = r"E:\AF3环境\ALL-DATA\DATA\ligands"        # 实验配体文件夹
OUT_DIR = r"E:\AF3环境\ALL-DATA\probe_comparison_exp"

os.makedirs(OUT_DIR, exist_ok=True)

def get_ca_list(pdb):
    st = gemmi.read_structure(pdb)
    out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.name == "CA":
                    out.append((chain.name, res.seqid.num, res.name, np.array([atom.pos.x, atom.pos.y, atom.pos.z])))
                    break
    return out

def match_seq(a, b):
    sa = [x[2] for x in a]
    sb = [x[2] for x in b]
    sm = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
    return [(blk.a + k, blk.b + k) for blk in sm.get_matching_blocks() for k in range(blk.size)]

def kabsch(P, Q):
    Pc = P.mean(0); Qc = Q.mean(0)
    P0 = P - Pc; Q0 = Q - Qc
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc

def get_true_center(lig_path):
    coords = []
    p = Path(lig_path); suf = p.suffix.lower()
    if suf in (".sdf", ".mol"):
        with open(p, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        n = int(lines[3][0:3].strip())
        for i in range(4, 4 + n):
            coords.append([float(lines[i][0:10]), float(lines[i][10:20]), float(lines[i][20:30])])
    elif suf == ".pdb":
        with open(p) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    elem = line[76:78].strip().upper() if len(line) > 76 else ""
                    if elem in ("H", "D"): continue
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.mean(coords, axis=0) if coords else None

def calc_probe_radius(pdb_path, center):
    st = gemmi.read_structure(pdb_path)
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()
    
    c = np.array(center, dtype=float)
    min_clearance = 5.0
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element.name in ("H", "D"): continue
                p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                d = np.linalg.norm(p - c)
                if d < 10.0:
                    clearance = d - vdw_of(atom.element.name)
                    if clearance < min_clearance:
                        min_clearance = clearance
    return max(min_clearance, 0.0)

def main():
    cif_files = glob.glob(os.path.join(CIF_DIR, "*", "*_MODEL.CIF"))
    results = []
    
    for idx, cif_path in enumerate(cif_files, 1):
        case_name = os.path.basename(os.path.dirname(cif_path))
        apo_pdb = os.path.join(APO_DIR, f"{case_name}_apo.pdb")
        relaxed_pdb = os.path.join(RELAXED_DIR, f"{case_name}_relaxed.pdb")
        
        exp_rec = os.path.join(EXP_DIR, case_name, "receptor.pdb")
        lig_file = None
        for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb"):
            p = os.path.join(LIG_DIR, case_name, nm)
            if os.path.exists(p):
                lig_file = p; break
                
        if not (os.path.exists(relaxed_pdb) and os.path.exists(exp_rec) and lig_file):
            continue
            
        try:
            # 1. 获取实验配体质心
            tc = get_true_center(lig_file)
            
            # 2. 对齐 AF3 和 实验受体，映射坐标
            ca_af3 = get_ca_list(apo_pdb)
            ca_exp = get_ca_list(exp_rec)
            pairs = match_seq(ca_af3, ca_exp)
            if len(pairs) < 10: continue
            P = np.array([ca_af3[i][3] for i, _ in pairs])
            Q = np.array([ca_exp[j][3] for _, j in pairs])
            R, t = kabsch(P, Q)
            # 映射到 AF3 坐标系
            tc_af3 = R.T @ tc - (R.T @ t)
            
            # 3. 测量
            r_apo = calc_probe_radius(apo_pdb, tc_af3)
            r_relaxed = calc_probe_radius(relaxed_pdb, tc_af3)
            
            results.append({
                "case": case_name, 
                "probe_apo_A": round(r_apo, 2), 
                "probe_relaxed_A": round(r_relaxed, 2), 
                "delta_A": round(r_relaxed - r_apo, 2)
            })
            if idx % 50 == 0: print(f"已处理 {idx} 个...")
        except Exception as e:
            pass

    # 输出统计
    if results:
        deltas = np.array([r["delta_A"] for r in results])
        r_apo = np.array([r["probe_apo_A"] for r in results])
        r_relaxed = np.array([r["probe_relaxed_A"] for r in results])
        
        print("\n" + "="*50)
        print(f"📊 基于【实验真实配体】的物理精修干预统计 (n={len(results)})")
        print("="*50)
        print(f"精修前中位探针半径: {np.median(r_apo):.2f} Å")
        print(f"精修后中位探针半径: {np.median(r_relaxed):.2f} Å")
        print(f"空腔显著打开 (Δ > +0.5 Å): {np.sum(deltas > 0.5)} 个")
        print("="*50)

if __name__ == "__main__":
    main()