import os
import glob
import csv
import numpy as np
import gemmi
import difflib
from pathlib import Path

VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "H": 1.20}
def vdw_of(elem): return VDW.get(elem.upper(), 1.70)

# ⚠️ 确保这些路径和你的真实目录一致
CIF_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3_workflow"
APO_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD"
DONE_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD-DONE"
DLP_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD-DLP"
EXP_DIR = r"E:\AF3环境\ALL-DATA\DATA\exp_structures"
LIG_DIR = r"E:\AF3环境\ALL-DATA\DATA\ligands"
OUT_DIR = r"E:\AF3环境\ALL-DATA\probe_final_comparison"

os.makedirs(OUT_DIR, exist_ok=True)

def get_ca_list(pdb):
    st = gemmi.read_structure(str(pdb))
    out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.name == "CA":
                    out.append((chain.name, res.seqid.num, res.name, np.array([atom.pos.x, atom.pos.y, atom.pos.z])))
                    break
    return out

def match_seq(a, b):
    sa = [x[2] for x in a]; sb = [x[2] for x in b]
    sm = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
    return [(blk.a + k, blk.b + k) for blk in sm.get_matching_blocks() for k in range(blk.size)]

def kabsch(P, Q):
    Pc = P.mean(0); Qc = Q.mean(0); P0 = P - Pc; Q0 = Q - Qc
    H = P0.T @ Q0; U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc

def get_true_center(lig_path):
    coords = []
    p = Path(lig_path); suf = p.suffix.lower()
    if suf in (".sdf", ".mol"):
        with open(p, encoding="utf-8", errors="ignore") as f: lines = f.readlines()
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

def calc_probe(pdb_path, center):
    if not os.path.exists(pdb_path):
        return None
    st = gemmi.read_structure(str(pdb_path))
    st.remove_alternative_conformations(); st.remove_hydrogens(); st.remove_waters()
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
    
    print(f"找到 {len(cif_files)} 个原始 CIF 文件，开始比对...")
    
    for idx, cif_path in enumerate(cif_files, 1):
        case_name = os.path.basename(os.path.dirname(cif_path))
        apo_pdb = os.path.join(APO_DIR, f"{case_name}_apo.pdb")
        done_pdb = os.path.join(DONE_DIR, f"{case_name}_relaxed.pdb")
        dlp_pdb = os.path.join(DLP_DIR, f"{case_name}_dlp.pdb")
        
        exp_rec = os.path.join(EXP_DIR, case_name, "receptor.pdb")
        lig_file = None
        for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb"):
            if os.path.exists(os.path.join(LIG_DIR, case_name, nm)):
                lig_file = os.path.join(LIG_DIR, case_name, nm)
                break
                
        if not (os.path.exists(apo_pdb) and os.path.exists(exp_rec) and lig_file):
            continue
            
        try:
            tc = get_true_center(lig_file)
            ca_af3 = get_ca_list(apo_pdb); ca_exp = get_ca_list(exp_rec)
            pairs = match_seq(ca_af3, ca_exp)
            if len(pairs) < 10: continue
            
            P = np.array([ca_af3[i][3] for i, _ in pairs])
            Q = np.array([ca_exp[j][3] for _, j in pairs])
            R, t = kabsch(P, Q)
            tc_af3 = R.T @ tc - (R.T @ t)
            
            r_af3 = calc_probe(apo_pdb, tc_af3)
            r_done = calc_probe(done_pdb, tc_af3)
            r_dlp = calc_probe(dlp_pdb, tc_af3)
            
            results.append({
                "case": case_name,
                "probe_af3": round(r_af3, 2) if r_af3 is not None else "",
                "probe_openmm": round(r_done, 2) if r_done is not None else "",
                "probe_dlp": round(r_dlp, 2) if r_dlp is not None else ""
            })
            if idx % 50 == 0: 
                print(f"已处理 {idx} 个...")
        except Exception as e:
            print(f"跳过 {case_name}: {e}")

    # 输出 CSV
    csv_path = os.path.join(OUT_DIR, "final_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "probe_af3", "probe_openmm", "probe_dlp"])
        w.writeheader(); w.writerows(results)

    # 终端统计
    if results:
        af3_arr = np.array([r["probe_af3"] for r in results if r["probe_af3"] != ""])
        done_arr = np.array([r["probe_openmm"] for r in results if r["probe_openmm"] != ""])
        dlp_arr = np.array([r["probe_dlp"] for r in results if r["probe_dlp"] != ""])
        
        print("\n" + "="*50)
        print(f"📊 全量探针对比统计 (n={len(results)})")
        print("="*50)
        print(f"1. AF3 原始结构中位半径 : {np.median(af3_arr):.2f} Å")
        if len(done_arr): print(f"2. OpenMM 精修中位半径: {np.median(done_arr):.2f} Å")
        if len(dlp_arr): print(f"3. DLPacker 重建中位半径: {np.median(dlp_arr):.2f} Å")
        
        # 计算修复率 (>1.5 Å 认为口袋被打开)
        if len(done_arr):
            opened_done = np.sum(done_arr > 1.5) - np.sum(af3_arr > 1.5)
            print(f"   OpenMM 新增打开 (>1.5Å) 数量: {opened_done} 个")
        if len(dlp_arr):
            opened_dlp = np.sum(dlp_arr > 1.5) - np.sum(af3_arr > 1.5)
            print(f"   DLPacker 新增打开 (>1.5Å) 数量: {opened_dlp} 个")
        print("="*50)

if __name__ == "__main__":
    main()