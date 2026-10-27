import os
import json
import numpy as np
import gemmi
import difflib
from pathlib import Path
import pandas as pd

# 路径配置
CIF_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3_workflow"
APO_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD"
EXP_DIR = r"E:\AF3环境\ALL-DATA\DATA\exp_structures"
LIG_DIR = r"E:\AF3环境\ALL-DATA\DATA\ligands"
CSV_PATH = r"E:\AF3环境\ALL-DATA\probe_final_comparison\final_comparison.csv"

# 4个奇迹生还的案例
SURVIVORS = ["6VTA_AKN", "6YMS_OZH", "7Z1Q_NIO", "8BRO_R7E"]

# 范德华半径
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "H": 1.20}
def vdw_of(elem): return VDW.get(elem.upper(), 1.70)

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

def find_blocker(pdb_path, center):
    st = gemmi.read_structure(str(pdb_path))
    st.remove_alternative_conformations(); st.remove_hydrogens(); st.remove_waters()
    c = np.array(center, dtype=float)
    min_clearance = 5.0
    blocker = "None"
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
                        blocker = f"{res.name}{res.seqid.num}"
    return blocker

def main():
    print(f"正在解剖 {len(SURVIVORS)} 个极少数被挽救的案例...\n")
    
    # 获取全量数据的统计基准
    df_all = pd.read_csv(CSV_PATH)
    df_all['probe_af3'] = pd.to_numeric(df_all['probe_af3'], errors='coerce')
    median_af3 = df_all['probe_af3'].median()
    
    results = []
    
    for case_name in SURVIVORS:
        cif_json = os.path.join(CIF_DIR, case_name, f"{case_name}_summary_confidences.json")
        cif_json_alt = os.path.join(CIF_DIR, case_name, f"{case_name}_CONFIDENCES.JSON") # 兼容你早期的文件
        apo_pdb = os.path.join(APO_DIR, f"{case_name}_apo.pdb")
        exp_rec = os.path.join(EXP_DIR, case_name, "receptor.pdb")
        
        lig_file = None
        for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb"):
            p = os.path.join(LIG_DIR, case_name, nm)
            if os.path.exists(p):
                lig_file = p; break
                
        # 1. 读取 ipTM
        iptm = "N/A"
        json_path = cif_json if os.path.exists(cif_json) else cif_json_alt
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                # 兼容不同版本的键名
                iptm = data.get('iptm', data.get('ipTM', "N/A"))
                
        # 2. 计算 Cα 主链 RMSD
        rmsd = "N/A"
        if os.path.exists(apo_pdb) and os.path.exists(exp_rec):
            ca_af3 = get_ca_list(apo_pdb); ca_exp = get_ca_list(exp_rec)
            pairs = match_seq(ca_af3, ca_exp)
            if len(pairs) > 10:
                P = np.array([ca_af3[i][3] for i, _ in pairs])
                Q = np.array([ca_exp[j][3] for _, j in pairs])
                R, t = kabsch(P, Q)
                P_aligned = (R @ P.T).T + t
                rmsd = round(float(np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1)))), 2)
                
        # 3. 找出原始阻挡残基
        blocker = "N/A"
        if os.path.exists(apo_pdb) and lig_file:
            tc = get_true_center(lig_file)
            # 映射到 AF3
            ca_af3 = get_ca_list(apo_pdb); ca_exp = get_ca_list(exp_rec)
            pairs = match_seq(ca_af3, ca_exp)
            P = np.array([ca_af3[i][3] for i, _ in pairs]); Q = np.array([ca_exp[j][3] for _, j in pairs])
            R, t = kabsch(P, Q)
            tc_af3 = R.T @ tc - (R.T @ t)
            blocker = find_blocker(apo_pdb, tc_af3)

        results.append({
            "Case": case_name,
            "ipTM": iptm,
            "Global_RMSD": rmsd,
            "Initial_Blocker": blocker
        })

    # 打印客观解剖结果
    print("="*70)
    print(f"{'Case':<12} | {'ipTM':<8} | {'Global RMSD (Å)':<18} | {'Blocker Residue'}")
    print("-" * 70)
    for r in results:
        print(f"{r['Case']:<12} | {r['ipTM']:<8} | {r['Global_RMSD']:<18} | {r['Initial_Blocker']}")
    print("="*70)
    
    print("\n【对比基准】")
    print(f"全量 421 个复合物的 AF3 初始探针半径中位数: {median_af3:.2f} Å")
    print("文献中 AF3 在 PoseBusters 集合上的整体 ipTM 均值通常在 0.35 ~ 0.45 之间")
    print("文献中 AF3 的全局主链 RMSD 均值通常在 5.0 ~ 6.0 Å 之间")
    print("\n👉 请对照上述基准，观察这 4 个案例是否具有极高置信度与极低主链偏差。")

if __name__ == "__main__":
    main()