import os, numpy as np, gemmi
import ligand_fit as L   # 复用你文件里的 get_true_center / find_lig / EXP_DIR

for case in ["5S8I_2LY", "5SD5_HWI", "5SB2_1K2"]:
    lig = L.find_lig(case)
    tc = L.get_true_center(lig)                    # Å
    exp = os.path.join(L.EXP_DIR, case, "receptor.pdb")
    st = gemmi.read_structure(exp)
    st.remove_hydrogens(); st.remove_waters()
    ds = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element.name in ("H","D"): continue
                p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                ds.append((float(np.linalg.norm(p - tc)),
                           f"{res.name}{res.seqid.num}:{atom.name}"))
    ds.sort()
    n5 = sum(1 for d,_ in ds if d < 5.0)
    print(f"{case}: tc 到 exp 最近重原子 {ds[0][0]:.2f} Å ({ds[0][1]}), 5Å内 {n5} 个原子")