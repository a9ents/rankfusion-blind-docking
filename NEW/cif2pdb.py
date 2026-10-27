import sys
import gemmi

def convert_cif_to_apo_pdb(cif_path, pdb_path):
    print(f"正在读取: {cif_path}")
    try:
        # 读取 AF3 的 CIF 文件
        st = gemmi.read_structure(cif_path)
    except Exception as e:
        print(f"读取失败，请检查文件路径: {e}")
        return

    # 遍历结构中的每条链、每个残基
    for model in st:
        for chain in model:
            to_remove = []
            for i, res in enumerate(chain):
                # 判断是否为标准聚合物（蛋白/核酸）
                # AF3 输出的配体（如 HWI）和结晶水，其 entity_type 不是 Polymer
                if res.entity_type != gemmi.EntityType.Polymer:
                    to_remove.append(i)
            
            # 从后往前删，避免删除时索引错位
            for idx in reversed(to_remove):
                del chain[idx]

    # 丢弃所有非标准残基后，可能有的链变空了，清理空链
    st.remove_empty_chains()
    
    # 确保保留第一号模型（AF3通常只输出一个最优模型）
    st.setup_entities()
    
    # 写出为 PDB 文件
    st.write_pdb(pdb_path)
    print(f"✅ 转换成功！纯蛋白已保存至: {pdb_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python cif2pdb.py <input.cif> <output.pdb>")
        print("例如: python cif2pdb.py 5SD5_HWI_MODEL.CIF 5SD5_apo.pdb")
        sys.exit(1)
        
    input_cif = sys.argv[1]
    output_pdb = sys.argv[2]
    convert_cif_to_apo_pdb(input_cif, output_pdb)