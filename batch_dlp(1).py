import os
import glob
from dlpacker import DLPacker

# 路径设置
APO_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD"
DLP_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD-DLP"
os.makedirs(DLP_DIR, exist_ok=True)

pdb_files = glob.glob(os.path.join(APO_DIR, "*_apo.pdb"))
print(f"找到 {len(pdb_files)} 个待处理文件。")

for idx, pdb_path in enumerate(pdb_files, 1):
    case_name = os.path.basename(pdb_path).replace("_apo.pdb", "")
    out_path = os.path.join(DLP_DIR, f"{case_name}_dlp.pdb")
    
    # 断点续跑：如果已经存在就跳过
    if os.path.exists(out_path):
        print(f"[{idx}/{len(pdb_files)}] 跳过 (已存在): {case_name}")
        continue
        
    print(f"[{idx}/{len(pdb_files)}] 正在处理: {case_name} ...")
    try:
        # 加载模型并重建
        dlp = DLPacker(pdb_path)
        dlp.reconstruct_protein(order='sequence', output_filename=out_path)
    except Exception as e:
        print(f"❌ 失败: {case_name}, 原因: {e}")

print("🎉 批量 DLPacker 重建完成！")