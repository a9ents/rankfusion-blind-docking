import os, csv, math, random, shutil
import numpy as np
import gemmi, difflib
from pathlib import Path
import openmm as mm
from openmm.app import PDBFile, Modeller, ForceField, Simulation, CutoffNonPeriodic, HBonds
from openmm import LangevinMiddleIntegrator, Platform, unit, CustomExternalForce
from pdbfixer import PDBFixer

# ================= 路径 =================
APO_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD"
EXP_DIR = r"E:\AF3环境\ALL-DATA\DATA\exp_structures"
LIG_DIR = r"E:\AF3环境\ALL-DATA\DATA\ligands"
OUT_DIR = r"E:\AF3环境\ALL-DATA\NO_BB_PACKING"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "no_backbone_results.csv")

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

# ================= 核心：固定主链，随机侧链重排 =================
def run_fixed_backbone_packing(apo_pdb, tc_af3, out_pdb, seeds=5):
    fixer = PDBFixer(filename=apo_pdb)
    fixer.findMissingResidues(); fixer.findMissingAtoms(); fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    tmp = out_pdb.replace('.pdb', '_tmp_fixed.pdb')
    with open(tmp, 'w') as f: PDBFile.writeFile(fixer.topology, fixer.positions, f)
    
    pdb = PDBFile(tmp)
    modeller = Modeller(pdb.topology, pdb.positions)
    ff = ForceField('amber14-all.xml', 'implicit/obc2.xml')
    system = ff.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometer, constraints=HBonds)
    
    # 主链全部固定
    restraint = CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter('k', 100.0 * unit.kilocalories_per_mole / unit.angstroms**2)
    restraint.addPerParticleParameter('x0'); restraint.addPerParticleParameter('y0'); restraint.addPerParticleParameter('z0')
    
    for atom in modeller.topology.atoms():
        if atom.element.name in ('H', 'D'): continue
        if atom.name in ['N', 'CA', 'C', 'O']: # 只固定主链
            pos = modeller.positions[atom.index]
            restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
    system.addForce(restraint)
    
    try: plat = Platform.getPlatformByName('CUDA')
    except Exception: plat = Platform.getPlatformByName('CPU')
    
    best_probe = -1.0
    best_pdb = None
    
    # 用多个种子随机打乱侧链，跑 MD
    for seed in range(seeds):
        random.seed(seed)
        positions = list(modeller.positions)
        for atom in modeller.topology.atoms():
            if atom.element.name in ('H', 'D'): continue
            if atom.name not in ['N', 'CA', 'C', 'O']:
                px, py, pz = positions[atom.index].value_in_unit(unit.nanometer)
                px += random.uniform(-0.15, 0.15) # 大约 1.5 Å 的随机扰动
                py += random.uniform(-0.15, 0.15)
                pz += random.uniform(-0.15, 0.15)
                positions[atom.index] = mm.Vec3(px, py, pz)
        
        # 修复报错点：每次循环新建 integrator
        integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
        sim = Simulation(modeller.topology, system, integrator, plat)
        sim.context.setPositions(positions)
        sim.minimizeEnergy(maxIterations=2000, tolerance=10*unit.kilojoule_per_mole/unit.nanometer)
        
        pos = sim.context.getState(getPositions=True).getPositions()
        temp_out = os.path.join(OUT_DIR, f"tmp_seed{seed}.pdb")
        with open(temp_out, 'w') as f: PDBFile.writeFile(sim.topology, pos, f)
        r = calc_probe(temp_out, tc_af3)
        
        if r > best_probe:
            best_probe = r
            best_pdb = temp_out
            print(f"    [Seed {seed}] 当前最佳探针: {best_probe:.2f} Å")
    
    # 把最好的那个结果复制成最终输出
    if best_pdb:
        shutil.copy(best_pdb, out_pdb)
        
    # 清理临时文件
    for seed in range(seeds):
        temp = os.path.join(OUT_DIR, f"tmp_seed{seed}.pdb")
        if os.path.exists(temp) and temp != best_pdb: os.remove(temp)
    if os.path.exists(tmp): os.remove(tmp)

# ================= 主流程 =================
if __name__ == "__main__":
    test_cases = ["5S8I_2LY", "5SD5_HWI", "5SB2_1K2"]
    print(f"测试样本 (不修主链，随机侧链构象搜索): {test_cases}\n")
    
    rows = []
    for case in test_cases:
        apo = os.path.join(APO_DIR, f"{case}_apo.pdb")
        exp = os.path.join(EXP_DIR, case, "receptor.pdb")
        lig = find_lig(case)
        if not (os.path.exists(apo) and os.path.exists(exp) and lig): continue

        tc = get_true_center(lig)
        ca_af3 = get_ca_list(apo); ca_exp = get_ca_list(exp)
        pairs = match_seq(ca_af3, ca_exp)
        if len(pairs) < 10: continue
        P = np.array([ca_af3[i][3] for i,_ in pairs]); Q = np.array([ca_exp[j][3] for _,j in pairs])
        R, t = kabsch(P, Q)
        tc_af3 = R.T @ (tc - t)
        
        r_orig = calc_probe(apo, tc_af3)
        glob_orig = calc_global_ca_rmsd(apo, exp)
        
        print(f"\n{case}: 原始探针 {r_orig:.2f} Å, 全局 RMSD {glob_orig:.2f} Å")
        
        out_final = os.path.join(OUT_DIR, f"{case}_fixed_bb.pdb")
        try:
            # 跑 5 个随机构象种子
            run_fixed_backbone_packing(apo, tc_af3, out_final, seeds=5)
            
            r_final = calc_probe(out_final, tc_af3)
            glob_final = calc_global_ca_rmsd(out_final, exp)
            
            g_final_str = f"{glob_final:.2f}" if glob_final is not None else "NA"
            rows.append({"case":case, "r_orig":round(r_orig,2), "r_final":round(r_final,2),
                         "glob_orig":round(glob_orig,2) if glob_orig is not None else None, 
                         "glob_final":round(glob_final,2) if glob_final is not None else None})
                         
            print(f"  最终搜索后结果: 探针 {r_final:.2f} Å, 全局 RMSD {g_final_str} Å")
        except Exception as e:
            print(f"  ❌ 失败: {e}")

    if rows:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["case","r_orig","r_final","glob_orig","glob_final"])
            w.writeheader(); w.writerows(rows)
        print(f"\n结果已保存至: {CSV_PATH}")