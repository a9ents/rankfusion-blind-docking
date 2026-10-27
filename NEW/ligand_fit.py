import os, csv, math, traceback
import numpy as np
import gemmi, difflib
from pathlib import Path
from collections import Counter
from openmm.app import PDBFile, Modeller, ForceField, Simulation, CutoffNonPeriodic, HBonds
from openmm import LangevinMiddleIntegrator, Platform, unit, CustomExternalForce
from pdbfixer import PDBFixer

# ================= 路径 =================
APO_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD"
EXP_DIR = r"E:\AF3环境\ALL-DATA\DATA\exp_structures"
LIG_DIR = r"E:\AF3环境\ALL-DATA\DATA\ligands"
OUT_DIR = r"E:\AF3环境\ALL-DATA\GENTLE_FIT_RELAX"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "gentle_fit_results.csv")

# ============ 参数 ============
K_PULL_LEVELS = [(500.0, 10000), (1500.0, 10000), (3000.0, 10000)]
K_ANCHOR   = 50.0        # [修复] 增强锚定力，防止全局 RMSD 恶化
STAGE1_TEMP = 250.0
STAGE1_USE_GB = True

# 多半径 ICP 扫描
ICP_RADII = [6.0, 8.0, 10.0, 12.0]
N_ICP_REFINE = 3

# 口袋开口力（阶段 2a）
R_OPEN_NM   = 0.50       # [优化] 从 0.40 扩大到 0.50 (5.0 Å)，给侧链更多空间
K_OPEN_SIDE = 15000.0    # [优化] 降低侧链力常数，配合软核公式防止 NaN
K_OPEN_MAIN = 1000.0     # 主链弱推
OPEN_STEPS  = 50000      # [优化] 100 ps，给侧链足够时间翻转

# 配体精细排斥（阶段 2b）
K_LIG      = 10000.0
R_LIG      = 0.55
EXCL_STEPS = 40000       # 80 ps

# 弛豫（阶段 3）
RELAX_STEPS = 50000      # [优化] 100 ps
K_GHOST    = 2000.0      # [新增] 阶段 3 幽灵占位力常数

VDW = {"C":1.70,"N":1.55,"O":1.52,"S":1.80,"P":1.80,
       "F":1.47,"CL":1.75,"BR":1.85,"I":1.98,"H":1.20}
def vdw_of(e): return VDW.get(e.upper(), 1.70)

def is_h(atom):
    if atom.element is not None and atom.element.name.upper() in ("H", "D"):
        return True
    nm = atom.name.upper()
    return nm.startswith(('H', 'D'))

def is_backbone(atom):
    return atom.name in ('N', 'CA', 'C', 'O')

# ================= 几何工具 =================
def calc_probe(pdb_path, center_ang):
    """[修复] 探针计算：移除 5.0 的硬截断，允许负值以反映 clash 深度"""
    st = gemmi.read_structure(str(pdb_path))
    st.remove_alternative_conformations(); st.remove_hydrogens(); st.remove_waters()
    c = np.array(center_ang, dtype=float)
    mc = float('inf') # 初始无穷大
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element.name in ("H","D"): continue
                p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                d = np.linalg.norm(p - c)
                if d < 15.0: # 扩大搜索半径
                    clash_depth = d - vdw_of(atom.element.name)
                    if clash_depth < mc:
                        mc = clash_depth
    # 如果完全没原子，返回 5.0；否则返回实际值（负数代表 clash，正数代表空腔）
    return mc if mc != float('inf') else 5.0 

def get_ca_list(pdb):
    st = gemmi.read_structure(str(pdb)); out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.name == "CA":
                    out.append((chain.name, res.seqid.num, res.name,
                                np.array([atom.pos.x, atom.pos.y, atom.pos.z])))
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

def closest_heavy_distance_ang(pdb, point_ang):
    st = gemmi.read_structure(str(pdb)); st.remove_hydrogens(); st.remove_waters()
    best = (float('inf'), ""); n5 = 0
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element.name in ("H","D"): continue
                p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                d = np.linalg.norm(p - point_ang)
                if d < best[0]:
                    best = (d, f"{res.name}{res.seqid.num}/{chain.name}:{atom.name}")
                if d < 5.0: n5 += 1
    return best[0], best[1], n5

def diagnose_pocket_from_pdb(pdb_path, tc_af3_ang, radius_ang=5.0):
    st = gemmi.read_structure(str(pdb_path)); st.remove_hydrogens(); st.remove_waters()
    res_counter = Counter(); n_atoms = 0
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element.name in ("H", "D"): continue
                p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                if np.linalg.norm(p - tc_af3_ang) < radius_ang:
                    n_atoms += 1
                    res_counter[f"{res.name}{res.seqid.num}/{chain.name}"] += 1
    print(f"  [诊断] 离配体中心 <{radius_ang}Å 的重原子: {n_atoms} 个")
    for k, v in res_counter.most_common(6):
        print(f"    {k}: {v} 个重原子")

# ================= 多半径 ICP =================
def build_pocket_targets_one(apo_pdb, exp_rec, tc_exp_ang, radius, n_refine=N_ICP_REFINE):
    ca_af3 = get_ca_list(apo_pdb); ca_exp = get_ca_list(exp_rec)
    pairs = match_seq(ca_af3, ca_exp)
    pocket_pairs = [(i, j) for i, j in pairs
                    if np.linalg.norm(ca_exp[j][3] - tc_exp_ang) < radius]
    if len(pocket_pairs) < 4: # [优化] 放宽到 4 个，拯救 6.0A 半径
        raise ValueError(f"口袋 CA 太少: {len(pocket_pairs)}")

    P = np.array([ca_af3[i][3] for i, _ in pocket_pairs])
    Q = np.array([ca_exp[j][3] for _, j in pocket_pairs])
    R, t = kabsch(P, Q)
    for _ in range(n_refine):
        P_fit = (R @ P.T).T + t
        res = np.linalg.norm(P_fit - Q, axis=1)
        med = float(np.median(res))
        mask = res < max(med * 2.0, 3.0)
        if mask.sum() < 4: break
        P = P[mask]; Q = Q[mask]
        R, t = kabsch(P, Q)
        pocket_pairs = [p for k, p in enumerate(pocket_pairs) if mask[k]]

    P_final = np.array([ca_af3[i][3] for i, _ in pocket_pairs])
    Q_final = np.array([ca_exp[j][3] for _, j in pocket_pairs])
    P_fit = (R @ P_final.T).T + t
    residuals = np.linalg.norm(P_fit - Q_final, axis=1)
    local_rmsd = float(np.sqrt(np.mean(residuals**2)))

    tc_kabsch = R.T @ (tc_exp_ang - t)
    target_table = {}
    per_ca_keys = []
    for k, (i, j) in enumerate(pocket_pairs):
        chain, resnum, resname, _ = ca_af3[i]
        tgt_ang = R.T @ (ca_exp[j][3] - t)
        key = (chain, str(resnum))
        target_table[key] = tgt_ang / 10.0
        per_ca_keys.append(key)
    return tc_kabsch, target_table, {
        'n_pocket_ca': len(pocket_pairs),
        'local_rmsd': local_rmsd,
        'radius': radius,
        'per_ca_residual': residuals.tolist(),
        'per_ca_keys': per_ca_keys,
    }

def build_pocket_targets_best(apo_pdb, exp_rec, tc_exp_ang, radii=ICP_RADII):
    best = None
    print("  [多半径 ICP 扫描]")
    for r in radii:
        try:
            tc, table, diag = build_pocket_targets_one(apo_pdb, exp_rec, tc_exp_ang, r)
            d0, atom0, _ = closest_heavy_distance_ang(apo_pdb, tc)
            print(f"    r={r:4.1f}Å: CA={diag['n_pocket_ca']:3d}, "
                  f"localRMSD={diag['local_rmsd']:5.2f}Å, "
                  f"tc 最近 {atom0} {d0:.2f}Å")
            
            # [优化] 评分函数：严厉惩罚高 local RMSD
            if diag['local_rmsd'] > 8.0:
                score = 1000.0 
            else:
                score = diag['local_rmsd'] + max(0, 2.0 - d0) * 3.0
                
            if best is None or score < best[0]:
                best = (score, tc, table, diag)
        except Exception as e:
            print(f"    r={r:4.1f}Å: 失败 ({e})")
    if best is None:
        raise ValueError("所有半径均失败")
    return best[1], best[2], best[3]

def compute_k_scale(keys, residuals):
    return {k: float(max(0.1, min(1.0, 1.0 - (r - 3.0) / 10.0)))
            for k, r in zip(keys, residuals)}

# ================= 力场工具 =================
def progressive_minimize(sim, stages=None):
    if stages is None:
        stages = [(500, 500.0), (1000, 100.0), (2000, 10.0)]
    for iters, tol in stages:
        sim.minimizeEnergy(maxIterations=iters,
                           tolerance=tol*unit.kilojoule_per_mole/unit.nanometer)

def add_pull_force(sys, modeller, target_table, per_ca_k_scale=None):
    f = CustomExternalForce("k_global * k_scale * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    f.addGlobalParameter('k_global', 500.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    f.addPerParticleParameter('x0'); f.addPerParticleParameter('y0'); f.addPerParticleParameter('z0')
    f.addPerParticleParameter('k_scale')
    n = 0
    for atom in modeller.topology.atoms():
        if is_h(atom) or atom.name != 'CA': continue
        key = (atom.residue.chain.id, str(atom.residue.id))
        if key in target_table:
            tgt = target_table[key]
            ks = 1.0 if per_ca_k_scale is None else per_ca_k_scale.get(key, 1.0)
            f.addParticle(atom.index, [float(tgt[0]), float(tgt[1]), float(tgt[2]), float(ks)])
            n += 1
    sys.addForce(f); return n

def add_anchor_force(sys, modeller, target_table, pos_ref_nm):
    f = CustomExternalForce("k_anchor * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    f.addGlobalParameter('k_anchor', K_ANCHOR * unit.kilojoules_per_mole / unit.nanometer**2)
    f.addPerParticleParameter('x0'); f.addPerParticleParameter('y0'); f.addPerParticleParameter('z0')
    n = 0
    for atom in modeller.topology.atoms():
        if is_h(atom) or atom.name != 'CA': continue
        key = (atom.residue.chain.id, str(atom.residue.id))
        if key not in target_table:
            xyz = pos_ref_nm[atom.index]
            f.addParticle(atom.index, [float(xyz[0]), float(xyz[1]), float(xyz[2])])
            n += 1
    sys.addForce(f); return n

def add_pocket_open_force(sys, modeller, tc_nm, pos_nm, r_open_nm=R_OPEN_NM,
                          k_side=K_OPEN_SIDE, k_main=K_OPEN_MAIN):
    """[修复] 引入 soft-core (r^2 + eps) 防止 r->0 时梯度爆炸产生 NaN"""
    f = CustomExternalForce(
        "step(R_open - r) * (R_open - r)^3 * k_local; "
        "r = sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2 + 0.001)") # 0.001 nm^2 软核
    f.addGlobalParameter('R_open', r_open_nm * unit.nanometer)
    f.addGlobalParameter('cx', float(tc_nm[0]) * unit.nanometer)
    f.addGlobalParameter('cy', float(tc_nm[1]) * unit.nanometer)
    f.addGlobalParameter('cz', float(tc_nm[2]) * unit.nanometer)
    f.addPerParticleParameter('k_local')
    n_side = n_main = 0
    for atom in modeller.topology.atoms():
        if is_h(atom): continue
        d = np.linalg.norm(pos_nm[atom.index] - tc_nm)
        if d > r_open_nm * 1.5:  # 缩小作用范围，避免推远端原子
            continue
        k = k_main if is_backbone(atom) else k_side
        f.addParticle(atom.index, [k * unit.kilojoules_per_mole / unit.nanometer**2])
        if is_backbone(atom): n_main += 1
        else: n_side += 1
    sys.addForce(f)
    return n_side, n_main

def add_ligand_exclusion_force(sys, modeller, tc_nm, r_lig_nm, k_lig):
    """[修复] 同样引入 soft-core"""
    f = CustomExternalForce(
        "k_lig * step(R_lig - r) * (R_lig - r)^2; "
        "r = sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2 + 0.001)")
    f.addGlobalParameter('k_lig', k_lig * unit.kilojoules_per_mole / unit.nanometer**2)
    f.addGlobalParameter('R_lig', r_lig_nm * unit.nanometer)
    f.addGlobalParameter('cx', float(tc_nm[0]) * unit.nanometer)
    f.addGlobalParameter('cy', float(tc_nm[1]) * unit.nanometer)
    f.addGlobalParameter('cz', float(tc_nm[2]) * unit.nanometer)
    for atom in modeller.topology.atoms():
        if is_h(atom): continue
        f.addParticle(atom.index)
    sys.addForce(f)

# ================= 核心流程 =================
def run_gentle_fit(apo_pdb, tc_af3_ang, target_table, out_pdb,
                   per_ca_k_scale=None, k_pull_levels=None):
    if k_pull_levels is None: k_pull_levels = K_PULL_LEVELS
    tc_af3_nm = np.array(tc_af3_ang) / 10.0

    fixer = PDBFixer(filename=apo_pdb)
    fixer.findMissingResidues(); fixer.findMissingAtoms(); fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    modeller = Modeller(fixer.topology, fixer.positions)
    ff = ForceField('amber14-all.xml', 'implicit/obc2.xml')
    ff_vac = ForceField('amber14-all.xml') if not STAGE1_USE_GB else None

    try: plat = Platform.getPlatformByName('CUDA')
    except Exception: plat = Platform.getPlatformByName('CPU')
    print(f"    平台: {plat.getName()}")

    # ---------- 阶段 0 ----------
    print("    [阶段 0] 纯物理场预松弛...")
    sys0 = ff.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
                           nonbondedCutoff=1.0*unit.nanometer,
                           constraints=HBonds, hydrogenMass=1.5*unit.amu)
    int0 = LangevinMiddleIntegrator(300*unit.kelvin, 10.0/unit.picosecond, 0.002*unit.picoseconds)
    sim0 = Simulation(modeller.topology, sys0, int0, plat)
    sim0.context.setPositions(modeller.positions)
    progressive_minimize(sim0)
    pos0 = sim0.context.getState(getPositions=True).getPositions()
    pos0_nm = np.asarray(pos0.value_in_unit(unit.nanometer))

    # ---------- 阶段 1：主链弹簧渐进 ----------
    print("    [阶段 1] 渐进主链弹簧...")
    ff1 = ff if STAGE1_USE_GB else ff_vac
    sys1 = ff1.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
                            nonbondedCutoff=1.0*unit.nanometer,
                            constraints=HBonds, hydrogenMass=1.5*unit.amu)
    n_pull = add_pull_force(sys1, modeller, target_table, per_ca_k_scale)
    n_anchor = add_anchor_force(sys1, modeller, target_table, pos0_nm)
    print(f"    弹簧 CA: {n_pull}, 锚定 CA: {n_anchor}")

    int1 = LangevinMiddleIntegrator(STAGE1_TEMP*unit.kelvin,
                                    10.0/unit.picosecond, 0.002*unit.picoseconds)
    sim1 = Simulation(modeller.topology, sys1, int1, plat)
    sim1.context.setPositions(pos0)
    progressive_minimize(sim1)
    for k_val, n_steps in k_pull_levels:
        sim1.context.setParameter('k_global',
                                  k_val * unit.kilojoules_per_mole / unit.nanometer**2)
        sim1.step(n_steps)
        print(f"      k_global={k_val:6.0f}, 跑 {n_steps} 步 ({n_steps*2/1000:.0f} ps)")
    pos1 = sim1.context.getState(getPositions=True).getPositions()
    pos1_nm = np.asarray(pos1.value_in_unit(unit.nanometer))

    # ---------- 阶段 2a：口袋开口 ----------
    print("    [阶段 2a] 口袋开口力（侧链强推 / 主链弱推）...")
    ff2a = ff if STAGE1_USE_GB else ff_vac
    sys2a = ff2a.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
                              nonbondedCutoff=1.0*unit.nanometer,
                              constraints=HBonds, hydrogenMass=1.5*unit.amu)
    add_pull_force(sys2a, modeller, target_table, per_ca_k_scale)
    add_anchor_force(sys2a, modeller, target_table, pos1_nm)
    n_side, n_main = add_pocket_open_force(sys2a, modeller, tc_af3_nm, pos1_nm)
    print(f"    开口力原子: 侧链 {n_side}, 主链 {n_main}")

    int2a = LangevinMiddleIntegrator(300*unit.kelvin,
                                     10.0/unit.picosecond, 0.002*unit.picoseconds)
    sim2a = Simulation(modeller.topology, sys2a, int2a, plat)
    sim2a.context.setPositions(pos1)
    sim2a.context.setParameter('k_global',
                               k_pull_levels[-1][0] * unit.kilojoules_per_mole / unit.nanometer**2)
    progressive_minimize(sim2a)
    sim2a.step(OPEN_STEPS)
    print(f"      跑 {OPEN_STEPS} 步 ({OPEN_STEPS*2/1000:.0f} ps)")
    pos2a = sim2a.context.getState(getPositions=True).getPositions()
    pos2a_nm = np.asarray(pos2a.value_in_unit(unit.nanometer))

    # ---------- 阶段 2b：配体精细排斥 ----------
    print("    [阶段 2b] 配体精细排斥...")
    ff2b = ff if STAGE1_USE_GB else ff_vac
    sys2b = ff2b.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
                              nonbondedCutoff=1.0*unit.nanometer,
                              constraints=HBonds, hydrogenMass=1.5*unit.amu)
    add_pull_force(sys2b, modeller, target_table, per_ca_k_scale)
    add_anchor_force(sys2b, modeller, target_table, pos2a_nm)
    add_ligand_exclusion_force(sys2b, modeller, tc_af3_nm, R_LIG, K_LIG)

    int2b = LangevinMiddleIntegrator(300*unit.kelvin,
                                     10.0/unit.picosecond, 0.002*unit.picoseconds)
    sim2b = Simulation(modeller.topology, sys2b, int2b, plat)
    sim2b.context.setPositions(pos2a)
    sim2b.context.setParameter('k_global',
                               k_pull_levels[-1][0] * unit.kilojoules_per_mole / unit.nanometer**2)
    progressive_minimize(sim2b)
    sim2b.step(EXCL_STEPS)
    print(f"      跑 {EXCL_STEPS} 步 ({EXCL_STEPS*2/1000:.0f} ps)")
    pos2b = sim2b.context.getState(getPositions=True).getPositions()

    # ---------- 阶段 3：撤力弛豫 + 幽灵占位 ----------
    print("    [阶段 3] 撤去主链外力，保留弱占位力防止疏水塌陷...")
    sys3 = ff.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
                           nonbondedCutoff=1.0*unit.nanometer,
                           constraints=HBonds, hydrogenMass=1.5*unit.amu)
    
    # [新增] 幽灵占位力：用较弱的排斥力撑住口袋，防止 GB 模型导致侧链塌陷
    add_ligand_exclusion_force(sys3, modeller, tc_af3_nm, r_lig_nm=0.45, k_lig=K_GHOST)

    int3 = LangevinMiddleIntegrator(300*unit.kelvin, 10.0/unit.picosecond, 0.002*unit.picoseconds)
    sim3 = Simulation(modeller.topology, sys3, int3, plat)
    sim3.context.setPositions(pos2b)
    progressive_minimize(sim3)
    sim3.step(RELAX_STEPS)
    print(f"      跑 {RELAX_STEPS} 步 ({RELAX_STEPS*2/1000:.0f} ps)")

    pos_final = sim3.context.getState(getPositions=True).getPositions()
    with open(out_pdb, 'w') as f:
        PDBFile.writeFile(sim3.topology, pos_final, f)

# ================= 主流程 =================
if __name__ == "__main__":
    test_cases = ["5S8I_2LY", "5SD5_HWI", "5SB2_1K2"]
    print(f"测试样本 (多半径 ICP + 软核开口力 + 幽灵占位): {test_cases}\n")

    rows = []
    for case in test_cases:
        apo = os.path.join(APO_DIR, f"{case}_apo.pdb")
        exp = os.path.join(EXP_DIR, case, "receptor.pdb")
        lig = find_lig(case)
        if not (os.path.exists(apo) and os.path.exists(exp) and lig):
            print(f"{case}: 缺文件，跳过"); continue

        print(f"\n{'='*60}\n{case}")

        tc = get_true_center(lig)
        tc_kabsch, target_table, diag = build_pocket_targets_best(apo, exp, tc)
        print(f"  选定: radius={diag['radius']}Å, CA={diag['n_pocket_ca']}, "
              f"localRMSD={diag['local_rmsd']:.2f} Å")

        tc_af3 = tc_kabsch
        d0, atom0, n0 = closest_heavy_distance_ang(apo, tc_af3)
        print(f"  tc: 最近 {atom0} d={d0:.2f}Å, <5Å {n0} 个重原子")
        diagnose_pocket_from_pdb(apo, tc_af3, radius_ang=5.0)

        per_ca_k_scale = compute_k_scale(diag['per_ca_keys'], diag['per_ca_residual'])

        r_orig = calc_probe(apo, tc_af3)
        glob_orig = calc_global_ca_rmsd(apo, exp)
        # [优化] 打印时区分正负，负数代表 clash
        print(f"  原始探针 {r_orig:.2f} Å {'(Clash!)' if r_orig < 0 else ''}, 全局 RMSD {glob_orig:.2f} Å")

        out_final = os.path.join(OUT_DIR, f"{case}_gentle_relaxed.pdb")
        try:
            run_gentle_fit(apo, tc_af3, target_table, out_final,
                           per_ca_k_scale=per_ca_k_scale)
            r_final = calc_probe(out_final, tc_af3)
            glob_final = calc_global_ca_rmsd(out_final, exp)
            g_final_str = f"{glob_final:.2f}" if glob_final is not None else "NA"

            print(f"  最终口袋:")
            diagnose_pocket_from_pdb(out_final, tc_af3, radius_ang=5.0)

            rows.append({
                "case": case, "radius": diag['radius'],
                "local_rmsd": round(diag['local_rmsd'], 2),
                "r_orig": round(r_orig, 2), "r_final": round(r_final, 2),
                "glob_orig": round(glob_orig, 2) if glob_orig is not None else None,
                "glob_final": round(glob_final, 2) if glob_final is not None else None,
            })
            print(f"  ✅ 最终松弛结果: 探针 {r_orig:.2f} → {r_final:.2f} Å, "
                  f"全局 RMSD {glob_orig:.2f} → {g_final_str} Å")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            traceback.print_exc()

    if rows:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["case","radius","local_rmsd",
                                              "r_orig","r_final","glob_orig","glob_final"])
            w.writeheader(); w.writerows(rows)
        print(f"\n结果已保存至: {CSV_PATH}")