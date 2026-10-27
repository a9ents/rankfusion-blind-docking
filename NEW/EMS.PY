import os
import glob

EXP_DIR = r"E:\AF3环境\ALL-DATA\DATA\exp_structures"
esm_cases = []
total = 0

for d in os.listdir(EXP_DIR):
    folder = os.path.join(EXP_DIR, d)
    if not os.path.isdir(folder): continue
    pdb = os.path.join(folder, "receptor.pdb")
    if os.path.exists(pdb):
        total += 1
        with open(pdb, 'r', errors='ignore') as f:
            head = f.read(2000).upper()
            if "ESMFOLD" in head or "ESM-METAGENOMIC" in head or ("PREDICTED" in head and "ALPHAFOLD" not in head):
                esm_cases.append(d)

print(f"实验集总数: {total}")
print(f"疑似 ESMFold 变体数量: {len(esm_cases)}")
print("清单:", esm_cases)