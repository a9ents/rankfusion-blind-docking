import os
import glob
import csv
import numpy as np
import gemmi
from collections import Counter

# 范德华半径（Å）
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "H": 1.20}
def vdw_of(elem): return VDW.get(elem.upper(), 1.70)

# 路径设置
CIF_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3_workflow"
APO_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD"
RELAXED_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD-DONE"
OUT_DIR = r"E:\AF3环境\ALL-DATA\probe_comparison"

os.makedirs(OUT_DIR, exist_ok=True)

def get_true_center_from_cif(cif_path):
    """从原始 CIF 中提取非聚合物（配体）的质心"""
    st = gemmi.read_structure(cif_path)
    coords = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.entity_type != gemmi.EntityType.Polymer and res.name != 'HOH':
                    for atom in res:
                        if atom.element.name not in ('H', 'D'):
                            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.mean(coords, axis=0) if coords else None

def calc_probe_radius(pdb_path, center):
    """计算给定 PDB 在 center 处的最大探针半径及最近阻挡原子"""
    st = gemmi.read_structure(pdb_path)
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()
    
    c = np.array(center, dtype=float)
    min_clearance = 5.0
    blocker_info = "None"
    
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element.name in ("H", "D"): continue
                p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                d = np.linalg.norm(p - c)
                
                # 只考虑 10 Å 内的原子
                if d < 10.0:
                    clearance = d - vdw_of(atom.element.name)
                    if clearance < min_clearance:
                        min_clearance = clearance
                        blocker_info = f"{res.name}{res.seqid.num}:{atom.name}"
                        
    return max(min_clearance, 0.0), blocker_info

def main():
    # 寻找所有 CIF 文件
    cif_files = glob.glob(os.path.join(CIF_DIR, "*", "*_MODEL.CIF"))
    print(f"找到 {len(cif_files)} 个 CIF 文件，开始对比分析...")
    
    results = []
    
    for idx, cif_path in enumerate(cif_files, 1):
        case_name = os.path.basename(os.path.dirname(cif_path)) # 比如 8G43_ZU6
        
        apo_pdb = os.path.join(APO_DIR, f"{case_name}_apo.pdb")
        relaxed_pdb = os.path.join(RELAXED_DIR, f"{case_name}_relaxed.pdb")
        
        # 如果精修文件还没生成，就先跳过
        if not os.path.exists(relaxed_pdb):
            continue
            
        print(f"[{idx}/{len(cif_files)}] 处理: {case_name}")
        
        try:
            center = get_true_center_from_cif(cif_path)
            if center is None:
                continue
                
            # 测量精修前
            r_apo, block_apo = calc_probe_radius(apo_pdb, center)
            # 测量精修后
            r_relaxed, block_relaxed = calc_probe_radius(relaxed_pdb, center)
            
            delta = r_relaxed - r_apo
            
            results.append({
                "case": case_name,
                "probe_apo_A": round(r_apo, 2),
                "probe_relaxed_A": round(r_relaxed, 2),
                "delta_A": round(delta, 2),
                "blocker_apo": block_apo,
                "blocker_relaxed": block_relaxed
            })
        except Exception as e:
            print(f"  ❌ 报错: {e}")

    # 保存 CSV
    csv_path = os.path.join(OUT_DIR, "relax_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "probe_apo_A", "probe_relaxed_A", "delta_A", "blocker_apo", "blocker_relaxed"])
        w.writeheader()
        w.writerows(results)
        
    print(f"\n✅ 分析完成！结果已保存至: {csv_path}")
    
    # 终端打印统计摘要
    if results:
        deltas = np.array([r["delta_A"] for r in results])
        r_apo = np.array([r["probe_apo_A"] for r in results])
        r_relaxed = np.array([r["probe_relaxed_A"] for r in results])
        
        print("\n" + "="*50)
        print(f"📊 物理精修干预统计 (共 {len(results)} 个复合物)")
        print("="*50)
        print(f"精修前中位探针半径: {np.median(r_apo):.2f} Å")
        print(f"精修后中位探针半径: {np.median(r_relaxed):.2f} Å")
        print(f"平均空腔增加值 (Δ): {np.mean(deltas):+.2f} Å")
        
        n_opened = np.sum(deltas > 0.5)
        n_closed = np.sum(deltas < -0.5)
        print(f"\n空腔被物理法则显著打开 (Δ > +0.5 Å): {n_opened} 个 ({n_opened/len(results)*100:.1f}%)")
        print(f"空腔反而变小 (Δ < -0.5 Å): {n_closed} 个 ({n_closed/len(results)*100:.1f}%)")
        print("="*50)

if __name__ == "__main__":
    main()