import os, csv, math
import numpy as np
import gemmi, difflib
from pathlib import Path
from openmm.app import PDBFile, Modeller, ForceField, Simulation, CutoffNonPeriodic, HBonds
from openmm import LangevinMiddleIntegrator, Platform, unit, CustomExternalForce
from pdbfixer import PDBFixer

# ================= 路径 =================
APO_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD"
EXP_DIR = r"E:\AF3环境\ALL-DATA\DATA\exp_structures"
LIG_DIR = r"E:\AF3环境\ALL-DATA\DATA\ligands"
OUT_DIR = r"E:\AF3环境\ALL-DATA\CRYO_WARM"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "cryo_warm_results.csv")

VDW = {"C":1.70,"N":1.55,"O":1.52,"S":1.80,"P":1.80,"F":1.47,"CL":1.75,"BR":1.85,"I":1.98,"H":1.20}
def vdw_of(e): return VDW.get(e.upper(), 1.70)

def calc_probe(pdb_path, center):
    st = gemmi.read_structure(str(pdb_path))
    st.remove_alternative_conformations(); st.remove_hydrogens(); st.remove_waters()
    c = np.array(center, dtype=float); mc = 5.0
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element.name in ("H","D"): continue
                p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                d = np.linalg.norm(p - c)
                if d < 10.0:
                    mc = min(mc, d - vdw_of(atom.element.name))
    return max(mc, 0.0)

def get_ca_list(pdb):
    st = gemmi.read_structure(str(pdb)); out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.name == "CA":
                    out.append((chain.name, res.seqid.num, res.name, np.array([atom.pos.x, atom.pos.y, atom.pos.z])))
                    break
    return out

def match_seq(a, b):
    sm = difflib.SequenceMatcher(None, [x[2] for x in a], [x[2] for x in b], autojunk=False)
    return [(blk.a+k, blk.b+k) for blk in sm.get_matching_blocks() for k in range(blk.size)]

def kabsch(P, Q):
    Pc = P.mean(0); Qc = Q.mean(0); P0 = P - Pc; Q0 = Q - Qc
    H = P0.T @ Q0; U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc

def get_true_center(lig_path):
    coords = []; p = Path(lig_path); suf = p.suffix.lower()
    if suf in (".sdf",".mol"):
        lines = open(p, encoding="utf-8", errors="ignore").readlines()
        n = int(lines[3][0:3].strip())
        for i in range(4, 4+n):
            coords.append([float(lines[i][0:10]), float(lines[i][10:20]), float(lines[i][20:30])])
    elif suf == ".pdb":
        for line in open(p):
            if line.startswith(("ATOM","HETATM")):
                elem = line[76:78].strip().upper() if len(line) > 76 else ""
                if elem in ("H","D"): continue
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.mean(coords, axis=0) if coords else None

def find_lig(case):
    for nm in ("ligand.sdf","ligand.mol","ligand.pdb"):
        p = os.path.join(LIG_DIR, case, nm)
        if os.path.exists(p): return p
    return None

def calc_global_ca_rmsd(pdb1, pdb2):
    a = get_ca_list(pdb1); b = get_ca_list(pdb2)
    pairs = match_seq(a, b)
    if len(pairs) < 3: return None
    P = np.array([a[i][3] for i,_ in pairs]); Q = np.array([b[j][3] for _,j in pairs])
    R, t = kabsch(P, Q); P2 = (R @ P.T).T + t
    return float(np.sqrt(np.mean(np.sum((P2-Q)**2, axis=1))))

# ================= 核心：冷冻 -> 回暖 =================
def run_cryo_warm(apo_pdb, out_pdb):
    fixer = PDBFixer(filename=apo_pdb)
    fixer.findMissingResidues(); fixer.findMissingAtoms(); fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    modeller = Modeller(fixer.topology, fixer.positions)
    
    ff = ForceField('amber14-all.xml', 'implicit/obc2.xml')
    system = ff.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometer, constraints=HBonds)
    
    # 极小步长防止高温爆炸 (1 fs)
    integrator = LangevinMiddleIntegrator(300*unit.kelvin, 5.0/unit.picosecond, 0.001*unit.picoseconds)
    try: plat = Platform.getPlatformByName('CUDA')
    except Exception: plat = Platform.getPlatformByName('CPU')
    
    sim = Simulation(modeller.topology, system, integrator, plat)
    sim.context.setPositions(modeller.positions)
    
    print("    [淬火] 初始能量最小化...")
    sim.minimizeEnergy(maxIterations=5000, tolerance=10*unit.kilojoule_per_mole/unit.nanometer)
    
    # 1. 极速冷却：300K -> 1K (10 ps)
    print("    [淬火] 极速冷却 300K -> 1K...")
    for temp in np.linspace(300, 1, 10):
        integrator.setTemperature(temp * unit.kelvin)
        sim.step(1000) # 每次 1 ps
        
    # 2. 冷冻维持 (20 ps)
    print("    [淬火] 低温 1K 冻结维持...")
    sim.step(20000)
    
    # 3. 缓慢回暖：1K -> 300K (50 ps)
    print("    [回暖] 缓慢回暖 1K -> 300K...")
    for temp in np.linspace(1, 300, 25):
        integrator.setTemperature(temp * unit.kelvin)
        sim.step(2000) # 每次 2 ps
        
    # 4. 常温最终弛豫 (10 ps)
    print("    [回暖] 300K 常温平衡...")
    integrator.setTemperature(300 * unit.kelvin)
    sim.step(10000)
    
    final_pos = sim.context.getState(getPositions=True).getPositions()
    with open(out_pdb, 'w') as f: PDBFile.writeFile(sim.topology, final_pos, f)

# ================= 主流程 =================
if __name__ == "__main__":
    test_cases = ["5S8I_2LY", "5SD5_HWI", "5SB2_1K2"]
    print(f"测试样本 (冷热循环法): {test_cases}\n")
    
    rows = []
    for case in test_cases:
        apo = os.path.join(APO_DIR, f"{case}_apo.pdb")
        exp = os.path.join(EXP_DIR, case, "receptor.pdb")
        lig = find_lig(case)
        if not (os.path.exists(apo) and os.path.exists(exp) and lig): continue

        tc = get_true_center(lig)
        ca_af3 = get_ca_list(apo); ca_exp = get_ca_list(exp)
        pairs = match_seq(ca_af3, ca_exp)
        P = np.array([ca_af3[i][3] for i,_ in pairs]); Q = np.array([ca_exp[j][3] for _,j in pairs])
        R, t = kabsch(P, Q)
        tc_af3 = R.T @ (tc - t)
        
        r_orig = calc_probe(apo, tc_af3)
        glob_orig = calc_global_ca_rmsd(apo, exp)
        
        print(f"\n{case}: 原始探针 {r_orig:.2f} Å, 全局 RMSD {glob_orig:.2f} Å")
        
        out_final = os.path.join(OUT_DIR, f"{case}_cryo_warm.pdb")
        try:
            run_cryo_warm(apo, out_final)
            
            r_final = calc_probe(out_final, tc_af3)
            glob_final = calc_global_ca_rmsd(out_final, exp)
            
            g_final_str = f"{glob_final:.2f}" if glob_final is not None else "NA"
            rows.append({"case":case, "r_orig":round(r_orig,2), "r_final":round(r_final,2),
                         "glob_orig":round(glob_orig,2) if glob_orig is not None else None, 
                         "glob_final":round(glob_final,2) if glob_final is not None else None})
                         
            print(f"  最终结果: 探针 {r_final:.2f} Å, 全局 RMSD {g_final_str} Å")
        except Exception as e:
            print(f"  ❌ 失败: {e}")

    if rows:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["case","r_orig","r_final","glob_orig","glob_final"])
            w.writeheader(); w.writerows(rows)
        print(f"\n结果已保存至: {CSV_PATH}")