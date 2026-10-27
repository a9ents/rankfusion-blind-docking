import os
import glob
import gemmi

# 定义你的源文件夹和目标文件夹
source_dir = r"E:\AF3环境\ALL-DATA\DATA\AF3_workflow"
target_dir = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD"

# 自动创建目标文件夹（如果不存在）
os.makedirs(target_dir, exist_ok=True)

# 递归查找所有以 _MODEL.CIF 结尾的文件
print(f"正在扫描文件夹: {source_dir} ...")
cif_files = glob.glob(os.path.join(source_dir, "*", "*_MODEL.CIF"))

if not cif_files:
    print("❌ 没有找到任何 *_MODEL.CIF 文件，请检查路径是否正确。")
    exit()

print(f"✅ 找到 {len(cif_files)} 个模型文件，开始批量转换...\n")

success_count = 0
fail_count = 0

for cif_path in cif_files:
    # 获取上一级文件夹的名字（例如 5SD5_HWI）
    folder_name = os.path.basename(os.path.dirname(cif_path))
    
    # 定义输出文件名
    output_name = f"{folder_name}_apo.pdb"
    output_path = os.path.join(target_dir, output_name)
    
    try:
        # 读取 CIF
        st = gemmi.read_structure(cif_path)
        
        # 剔除配体和水
        for model in st:
            for chain in model:
                to_remove = []
                for i, res in enumerate(chain):
                    if res.entity_type != gemmi.EntityType.Polymer:
                        to_remove.append(i)
                for idx in reversed(to_remove):
                    del chain[idx]
                    
        st.remove_empty_chains()
        st.setup_entities()
        
        # 保存为 PDB
        st.write_pdb(output_path)
        print(f"[成功] {folder_name}")
        success_count += 1
        
    except Exception as e:
        print(f"[失败] {folder_name} - 原因: {e}")
        fail_count += 1

print(f"\n🎉 批量转换完成！成功: {success_count} 个，失败: {fail_count} 个。")
print(f"所有纯蛋白 PDB 已保存在: {target_dir}")