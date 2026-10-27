import numpy as np
import gemmi
import os

# 范德华半径
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "H": 1.20}
def vdw_of(elem): return VDW.get(elem.upper(), 1.70)

def calc_probe(pdb_path, center):
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
                if d < 10.0:
                    clearance = d - vdw_of(atom.element.name)
                    if clearance < min_clearance:
                        min_clearance = clearance
                        blocker_info = f"{res.name}{res.seqid.num}:{atom.name}"
    return max(min_clearance, 0.0), blocker_info

# 假设你已经通过之前的脚本算出了 tc_af3（这个案例的真实口袋在 AF3 坐标系下的中心）
# 如果你忘了，这里我直接拿你之前跑的数据。8G43_ZU6 在 AF3-PBD-DONE 里的探针是 0.14 Å。
# 为了演示，你需要把 tc_af3 填进来，或者我写一段代码直接从 CIF 里取（因为我们只是粗略验证，直接用 CIF 的非聚合物质心即可）
def get_true_center_from_cif(cif_path):
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

# 路径配置
case_name = "8G43_ZU6"
cif_path = f"DATA/AF3_workflow/{case_name}/{case_name}_MODEL.CIF"
af3_pdb = f"DATA/AF3-PBD/{case_name}_apo.pdb"
dlp_pdb = f"{case_name}_dlp_repacked.pdb"

if __name__ == "__main__":
    center = get_true_center_from_cif(cif_path)
    
    r_af3, b_af3 = calc_probe(af3_pdb, center)
    r_dlp, b_dlp = calc_probe(dlp_pdb, center)
    
    print("\n" + "="*40)
    print(f"📊 {case_name} 空腔探针验证")
    print("="*40)
    print(f"AF3 原始结构探针半径: {r_af3:.2f} Å (阻挡残基: {b_af3})")
    print(f"DLPacker 重建探针半径: {r_dlp:.2f} Å (阻挡残基: {b_dlp})")
    print(f"空腔变化 (Delta):    {r_dlp - r_af3:+.2f} Å")
    
    if r_dlp > 1.5 and r_af3 < 1.0:
        print("\n🎉 完美！DLPacker 成功打开了被 AF3 堵死的口袋！")
    elif r_dlp > r_af3 + 0.5:
        print("\n✅ 有改善！DLPacker 把侧链移开了一部分。")
    else:
        print("\n⚠️ 改善不明显。说明 AF3 的主链本身可能已经扭曲，单纯重建侧链救不回来。")