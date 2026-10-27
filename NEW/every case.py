# -*- coding: utf-8 -*-
"""
配体引导插入 + 四层热力学验证 (v13.0 修复版 - 仅跑指定4个case)
=====================================================================
v13.0 修复：
  [FIX-1] pre_filter_backbone 返回安全的 dict（含 rama_frac/angle_bad），
          process_case 使用 .get() 防止 KeyError。
  [FIX-2] SCALE_SCHEDULE 前几级放大，确保插入力从第 1 级就生效
          （避免 opened.pdb == apo.pdb 的死局）。
  [FIX-3] L3 采样范围自适应：先跑短 MD 探测 s，再决定 PMF 范围。
  [FIX-4] L1 回弹记录每个 monitor 点，确保 end 是真正的终点。
  [FIX-5] geom_clash_before/after 从 steered_insertion 真实传回。
"""
import os, glob, csv, traceback, json, sys, shutil
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import gemmi
import difflib

from openmm.app import (
    PDBFile, Modeller, ForceField, Simulation, CutoffNonPeriodic, HBonds
)
from openmm import (
    LangevinMiddleIntegrator, Platform, unit,
    CustomExternalForce, CustomCVForce, Vec3
)
from pdbfixer import PDBFixer


# ==================== 路径 ====================
AF3_DIR   = r"E:\AF3环境\ALL-DATA\DATA\AF3_workflow"
EXP_DIR   = r"E:\AF3环境\ALL-DATA\DATA\exp_structures"
LIG_DIR   = r"E:\AF3环境\ALL-DATA\DATA\ligands"
OUT_DIR   = r"E:\AF3环境\ALL-DATA\SMD_REPAIR_RESULTS"
os.makedirs(OUT_DIR, exist_ok=True)

# 只跑这 4 个
TARGET_CASES = ["6XBO_5MC", "7BLA_WCS", "7MY1_IPE", "7N4W_P4V"]


# ==================== 参数 ====================
K_INSERT          = 10.0
N_INSERT_STAGES   = 23
MIN_PER_STAGE     = 2000
K_POS_FINAL       = 50.0

# [FIX-2] 放大前几级，确保 R0 * scale > 0.1 Å（正则化下限）
SCALE_SCHEDULE = [
    0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.36,
    0.42, 0.48, 0.54, 0.60, 0.66, 0.72, 0.78, 0.83,
    0.88, 0.92, 0.95, 0.97, 0.985, 0.995, 1.0
]
assert len(SCALE_SCHEDULE) == N_INSERT_STAGES

GEOM_CLASH_FACTOR    = 0.4
GEOM_MAX_ITER        = 30
GEOM_REJECT_THRESH   = 5
GEOM_MIN_STEP        = 0.05
GEOM_KICK_SCALE      = 0.5

ENABLE_PREHEAT        = True
PREHEAT_CLASH_THRESH  = 80
PREHEAT_TEMP          = 500 * unit.kelvin
PREHEAT_STEPS         = 250000
PREHEAT_SHELL_A       = 10.0
PREHEAT_K_OUTSIDE     = 5000.0

FF_FILES           = ("amber14-all.xml", "implicit/gbn2.xml")
NONBONDED_CUTOFF   = 1.0 * unit.nanometer
TEMPERATURE        = 300 * unit.kelvin
FRICTION           = 5.0 / unit.picosecond
TIMESTEP           = 0.002 * unit.picoseconds
RNG_SEED           = 20241022
REQUIRED_PLATFORM  = "CUDA"

VERIFY_CONFIG = {
    "RUN_L1": True, "RUN_L2": True, "RUN_L3": True, "RUN_L4_LITE": True,
    "L1_STEPS": 250_000, "L1_MONITOR": 2000,
    "L3_CV_R0": 6.0, "L3_CV_SIGMA": 0.3,
    "L3_N_BINS": 50,
    "L3_UPDATE_EVERY": 200,
    "L3_TOTAL_STEPS": 2_000_000,
    "L3_PROBE_STEPS": 5000,     # [FIX-3] 探测 s 范围
    "L3_RANGE_MARGIN": 3.0,     # [FIX-3]
    "L4_STEPS": 500_000, "L4_MONITOR": 5000,
    "L4_K_BACKBONE": 100.0,
}

VDW = {"C":1.70,"N":1.55,"O":1.52,"S":1.80,"P":1.80,"F":1.47,
       "CL":1.75,"BR":1.85,"I":1.98,"H":1.20}
def vdw_of(e): return VDW.get(e.upper(), 1.70)
BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}

CSV_KEYS = [
    "case","seed","seed_probe","seed_clash","seed_init_clash",
    "pre_filtered","filter_rama","filter_angle_bad",
    "geom_clash_before","geom_clash_after","geom_rejected",
    "preheated",
    "L1_start","L1_end","L1_drop","L1_verdict",
    "L2_rama","L2_clash","L2_rmsd","L2_verdict",
    "L3_F_range_kJ","L3_verdict",
    "L4_start","L4_end","L4_drop","L4_verdict",
    "error"
]


# ==================== 单位与坐标辅助 ====================
def k_quantity(kcal_per_mol_A2: float):
    return kcal_per_mol_A2 * unit.kilocalories_per_mole / unit.angstrom ** 2

def _to_numpy_positions(positions):
    if hasattr(positions, "value_in_unit"):
        vals = positions.value_in_unit(unit.nanometer)
    else:
        vals = positions
    try:
        arr = np.asarray(vals, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3: return arr
    except (ValueError, TypeError): pass
    return np.array([[float(v[0]), float(v[1]), float(v[2])] for v in vals], dtype=float)

def _atom_pos_nm(positions, idx):
    p = positions[idx]
    if hasattr(p, "value_in_unit"): v = p.value_in_unit(unit.nanometer)
    else: v = p
    return np.array([float(v[0]), float(v[1]), float(v[2])])

def _count_hydrogens(topology):
    n = 0
    for a in topology.atoms():
        el = a.element
        if el is not None and getattr(el, "symbol", None) in ("H", "D"): n += 1
        elif el is not None and getattr(el, "name", None) in ("H", "D"): n += 1
        elif a.name.startswith("H"): n += 1
    return n

def _max_force(sim):
    try:
        st = sim.context.getState(getForces=True)
        f = np.asarray(st.getForces().value_in_unit(unit.kilocalories_per_mole / unit.angstrom), dtype=float)
        return float(np.max(np.abs(f)))
    except Exception: return float("inf")

def _restore_positions(sim, pos_arr_nm):
    vecs = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in pos_arr_nm]
    sim.context.setPositions(vecs * unit.nanometer)

def safe_write_pdb(sim, path, fallback_positions_nm=None):
    try:
        pos = sim.context.getState(getPositions=True).getPositions()
        arr = _to_numpy_positions(pos)
        if not np.all(np.isfinite(arr)):
            if fallback_positions_nm is not None:
                _restore_positions(sim, fallback_positions_nm)
            else:
                return False
        with open(path, "w") as f:
            PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), f)
        return True
    except ValueError as e:
        if "NaN" in str(e): return False
        raise


# ==================== 几何清理 ====================
def detect_severe_clashes(coords_A, elems, factor=GEOM_CLASH_FACTOR):
    n = len(coords_A)
    if n == 0: return 0, []
    rad = np.array([vdw_of(e) for e in elems]) * factor
    conflicts = []
    B = 500
    for i0 in range(0, n, B):
        i1 = min(i0 + B, n)
        d = np.sqrt(((coords_A[i0:i1, None, :] - coords_A[None, :, :]) ** 2).sum(-1))
        thr = rad[i0:i1, None] + rad[None, :]
        close = d < thr
        close &= (np.arange(n)[None, :] > np.arange(i0, i1)[:, None])
        idx = np.argwhere(close)
        for a, b in idx: conflicts.append((int(a + i0), int(b)))
    return len(conflicts), conflicts


def clean_initial_clashes(coords_A, elems,
                           factor=GEOM_CLASH_FACTOR, max_iter=GEOM_MAX_ITER,
                           min_step=GEOM_MIN_STEP, kick_scale=GEOM_KICK_SCALE):
    coords = coords_A.copy()
    n_before, _ = detect_severe_clashes(coords, elems, factor)
    if n_before == 0: return coords, 0, 0, True

    print(f"  [几何清理] 初始冲突 {n_before} 对，开始迭代...")
    for it in range(1, max_iter + 1):
        n_now, conflicts = detect_severe_clashes(coords, elems, factor)
        if n_now == 0:
            return coords, n_before, 0, True
        for ai, aj in conflicts:
            v = coords[ai] - coords[aj]
            d = np.linalg.norm(v)
            if d < 1e-6:
                v = np.random.randn(3); d = np.linalg.norm(v)
            v = v / d
            step = max(min_step, (vdw_of(elems[ai]) + vdw_of(elems[aj])) * factor - d) * kick_scale
            coords[ai] += v * step * 0.5
            coords[aj] -= v * step * 0.5

    n_after, _ = detect_severe_clashes(coords, elems, factor)
    return coords, n_before, n_after, n_after == 0


def apply_coords_to_modeller(modeller, coords_A):
    pos_nm = coords_A * 0.1
    vecs = [Vec3(float(p[0]), float(p[1]), float(p[2])) for p in pos_nm]
    modeller.positions = vecs * unit.nanometer


def clean_pdb_geometry(pdb_path, out_path, label=""):
    print(f"\n[几何清理{' ' + label if label else ''}] {os.path.basename(pdb_path)}")
    pdb = PDBFile(pdb_path)
    modeller = Modeller(pdb.topology, pdb.positions)
    coords_nm = _to_numpy_positions(modeller.positions)
    coords_A = coords_nm * 10.0
    elems = [a.element.name if a.element is not None else "C"
             for a in modeller.topology.atoms()]

    if len(coords_A) != len(elems): return 0, 0, False

    new_coords, n_before, n_after, ok = clean_initial_clashes(coords_A, elems)
    apply_coords_to_modeller(modeller, new_coords)

    with open(out_path, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    return n_before, n_after, ok


# ==================== 基础工具 ====================
_platform_cache = None
def pick_platform():
    global _platform_cache
    if _platform_cache is not None: return _platform_cache
    avail = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    if REQUIRED_PLATFORM not in avail:
        raise RuntimeError(f"要求平台 {REQUIRED_PLATFORM!r} 不可用；已安装 {avail}。")
    _platform_cache = Platform.getPlatformByName(REQUIRED_PLATFORM)
    return _platform_cache

def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    P0, Q0 = P - Pc, Q - Qc
    U, S, Vt = np.linalg.svd(P0.T @ Q0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc

def get_ca_list(st):
    out = []
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

def strip_nonpolymer(st):
    st2 = st.clone()
    for model in st2:
        for chain in model:
            to_del = [i for i, r in enumerate(chain) if r.entity_type != gemmi.EntityType.Polymer]
            for i in reversed(to_del): del chain[i]
    return st2

def find_one_cif(d):
    for pat in ("*.cif", "*.cif.gz"):
        c = sorted(glob.glob(os.path.join(d, pat)))
        if c: return c[0]
    return None

def calc_probe_st(st_prot, center):
    c = np.asarray(center, dtype=float); mc = 5.0
    for chain in st_prot[0]:
        for res in chain:
            for atom in res:
                if atom.element.name in ("H","D"): continue
                p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                d = np.linalg.norm(p - c)
                if d < 10.0: mc = min(mc, d - vdw_of(atom.element.name))
    return max(mc, 0.0)

def count_clashes_st(st, ignore_sep=3, factor=0.55):
    coords, elems, chains, residx = [], [], [], []
    for chain in st[0]:
        for i, res in enumerate(chain):
            for atom in res:
                if atom.element.name in ("H","D"): continue
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                elems.append(atom.element.name); chains.append(chain.name); residx.append(i)
    coords = np.asarray(coords); n = len(coords)
    if n == 0: return 0
    rad = np.array([vdw_of(e) for e in elems]) * factor
    count = 0; B = 500
    chains = np.asarray(chains); residx = np.asarray(residx)
    for i0 in range(0, n, B):
        i1 = min(i0+B, n)
        d = np.sqrt(((coords[i0:i1,None,:] - coords[None,:,:])**2).sum(-1))
        close = d < rad[i0:i1,None] + rad[None,:]
        close &= (np.arange(n)[None,:] > np.arange(i0,i1)[:,None])
        same = chains[i0:i1,None] == chains[None,:]
        near = np.abs(residx[i0:i1,None] - residx[None,:]) <= ignore_sep
        close &= ~(same & near)
        count += int(close.sum())
    return count

def ca_rmsd_after_align(st_a, st_b):
    ca_a, ca_b = get_ca_list(st_a), get_ca_list(st_b)
    pairs = match_seq(ca_a, ca_b)
    if len(pairs) < 10: return None
    P = np.array([ca_a[i][3] for i, _ in pairs])
    Q = np.array([ca_b[j][3] for _, j in pairs])
    R, t = kabsch(P, Q)
    return float(np.sqrt((((R @ P.T).T + t - Q)**2).sum(1).mean()))


# ==================== PDBFixer ====================
def fix_pdb_with_fixer(pdb_in, pdb_out, label=""):
    print(f"\n[PDBFixer{' ' + label if label else ''}] 处理 {os.path.basename(pdb_in)}")
    try: fixer = PDBFixer(filename=pdb_in)
    except Exception as e: print(f"  ❌ 加载失败: {e}"); return False
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    with open(pdb_out, "w") as f: PDBFile.writeFile(fixer.topology, fixer.positions, f)
    print(f"  加氢后: {fixer.topology.getNumAtoms()} 原子, {_count_hydrogens(fixer.topology)} 个氢")
    return True


# ==================== ProbeTracker ====================
class ProbeTracker:
    def __init__(self): self.history: List[Tuple[str, float]] = []
    def record(self, label, topology, positions_nm, center_A):
        c = np.asarray(center_A, dtype=float)
        pos_arr = _to_numpy_positions(positions_nm)
        mc = 5.0
        for atom in topology.atoms():
            if atom.element.name in ("H", "D"): continue
            p = pos_arr[atom.index] * 10.0
            d = np.linalg.norm(p - c)
            if d < 10.0: mc = min(mc, d - vdw_of(atom.element.name))
        probe = max(mc, 0.0)
        self.history.append((label, probe))
        return probe
    def save_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("stage,probe_A\n")
            for s, p in self.history: f.write(f"{s},{p:.3f}\n")


# ==================== 配体读取 & Seed ====================
def read_ligand_heavy(lig_path):
    coords, elems = [], []
    p = Path(lig_path); suf = p.suffix.lower()
    if suf in (".sdf", ".mol"):
        lines = open(p, encoding="utf-8", errors="ignore").readlines()
        n = int(lines[3][0:3].strip())
        for i in range(4, 4+n):
            ln = lines[i]
            x, y, z = float(ln[0:10]), float(ln[10:20]), float(ln[20:30])
            e = ln[31:34].strip().upper() if len(ln) > 34 else ""
            if e in ("H", "D"): continue
            coords.append([x, y, z]); elems.append(e)
    elif suf == ".pdb":
        for line in open(p):
            if line.startswith(("ATOM","HETATM")):
                e = line[76:78].strip().upper() if len(line) > 76 else ""
                if e in ("H","D"): continue
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                elems.append(e)
    return np.asarray(coords, dtype=float), elems

def find_best_seed(case_name):
    exp_pdb = os.path.join(EXP_DIR, case_name, "receptor.pdb")
    lig_file = None
    for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb"):
        p = os.path.join(LIG_DIR, case_name, nm)
        if os.path.exists(p): lig_file = p; break
    if not (os.path.exists(exp_pdb) and lig_file): return None, None
    st_exp = gemmi.read_structure(exp_pdb)
    ca_exp = get_ca_list(st_exp)
    lig_coords_exp, lig_elems = read_ligand_heavy(lig_file)
    tc_exp = lig_coords_exp.mean(0)
    sample_dirs = sorted(glob.glob(os.path.join(AF3_DIR, case_name, "SEED-*_SAMPLE-*")))
    if not sample_dirs: return None, None
    best = None
    for s_dir in sample_dirs:
        cif = find_one_cif(s_dir)
        if not cif: continue
        try: st_af3 = gemmi.read_structure(cif)
        except: continue
        st_prot = strip_nonpolymer(st_af3)
        ca_af3 = get_ca_list(st_af3)
        pairs = match_seq(ca_af3, ca_exp)
        if len(pairs) < 10: continue
        P = np.array([ca_af3[i][3] for i,_ in pairs])
        Q = np.array([ca_exp[j][3] for _,j in pairs])
        R, t = kabsch(P, Q)
        lig_coords_af3 = (R.T @ (lig_coords_exp - t).T).T
        tc_af3 = R.T @ (tc_exp - t)
        probe = calc_probe_st(st_prot, tc_af3)
        n_cl = count_clashes_st(st_prot)
        init_clash = 0
        for chain in st_prot[0]:
            for res in chain:
                for atom in res:
                    if atom.element.name in ("H","D"): continue
                    p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    for lc, le in zip(lig_coords_af3, lig_elems):
                        if np.linalg.norm(p - lc) < 0.6*(vdw_of(atom.element.name) + vdw_of(le)):
                            init_clash += 1; break
        rec = {"cif": cif, "tc_af3": tc_af3, "lig_af3": lig_coords_af3,
               "lig_elems": lig_elems, "probe": probe, "clash": n_cl,
               "init_clash": init_clash, "dir": os.path.basename(s_dir)}
        if best is None: best = rec
        elif rec["clash"] <= 50 and (best["clash"] > 50 or rec["probe"] > best["probe"] + 0.2): best = rec
    if best is None: return None, None
    return best["cif"], best


# ==================== 平铺预处理 ====================
def preheat_unfold(pdb_in, pocket_center_A, platform_obj, out_pdb):
    print(f"\n[平铺预处理] T={PREHEAT_TEMP}, {PREHEAT_STEPS} 步")
    pdb = PDBFile(pdb_in)
    modeller = Modeller(pdb.topology, pdb.positions)
    ff = ForceField(*FF_FILES)
    system = ff.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
                             nonbondedCutoff=NONBONDED_CUTOFF, constraints=HBonds)
    lock = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    lock.addPerParticleParameter("k"); lock.addPerParticleParameter("x0")
    lock.addPerParticleParameter("y0"); lock.addPerParticleParameter("z0")
    c_A = np.asarray(pocket_center_A, dtype=float)
    pos_arr = _to_numpy_positions(modeller.positions)
    for atom in modeller.topology.atoms():
        if atom.element.name in ("H", "D"): continue
        p_A = pos_arr[atom.index] * 10.0
        if np.linalg.norm(p_A - c_A) >= PREHEAT_SHELL_A:
            p_nm = pos_arr[atom.index]
            lock.addParticle(atom.index, [k_quantity(PREHEAT_K_OUTSIDE), p_nm[0], p_nm[1], p_nm[2]])
    system.addForce(lock)
    integrator = LangevinMiddleIntegrator(PREHEAT_TEMP, FRICTION, TIMESTEP)
    integrator.setRandomNumberSeed(RNG_SEED)
    sim = Simulation(modeller.topology, system, integrator, platform_obj)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=2000)
    n_chunks = 10; per = max(1, PREHEAT_STEPS // n_chunks)
    for i in range(n_chunks): sim.step(per)
    integrator.setTemperature(TEMPERATURE); sim.step(50000)
    if not safe_write_pdb(sim, out_pdb): raise RuntimeError("平铺预处理输出 NaN")
    return out_pdb


# ==================== 配体引导插入 ====================
def steered_insertion(apo_pdb, lig_coords_A, lig_elems, pocket_center_A,
                      platform_obj, tracker, out_pdb, seed_info=None):
    print(f"\n[配体引导插入] {len(lig_coords_A)} 个配体球")

    cleaned_pdb = out_pdb.replace(".pdb", "_geomclean.pdb")
    n_before, n_after, geom_ok = clean_pdb_geometry(apo_pdb, cleaned_pdb)

    if not geom_ok:
        if n_after > GEOM_REJECT_THRESH:
            print(f"  ⛔ 清理后仍有 {n_after} 个严重冲突，拒绝")
            return None, n_before, n_after
        print(f"  ⚠ 清理后残留 {n_after} 个冲突，继续尝试")

    pdb = PDBFile(cleaned_pdb)
    modeller = Modeller(pdb.topology, pdb.positions)

    if _count_hydrogens(modeller.topology) == 0:
        tmp_fixed = out_pdb.replace(".pdb", "_tmp_fixed.pdb")
        if not fix_pdb_with_fixer(cleaned_pdb, tmp_fixed, label="(自动)"):
            raise RuntimeError("PDBFixer 失败")
        pdb = PDBFile(tmp_fixed); modeller = Modeller(pdb.topology, pdb.positions)

    ff = ForceField(*FF_FILES)
    system = ff.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
                             nonbondedCutoff=NONBONDED_CUTOFF, constraints=HBonds)

    init_clash = seed_info.get("init_clash", 0) if isinstance(seed_info, dict) else 0
    k_pos_start = 50.0 if init_clash > 50 else (200.0 if init_clash > 10 else 500.0)

    pos_restraint = CustomExternalForce("0.5 * k_pos * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    pos_restraint.addGlobalParameter("k_pos", k_pos_start)
    pos_restraint.addPerParticleParameter("x0"); pos_restraint.addPerParticleParameter("y0"); pos_restraint.addPerParticleParameter("z0")
    for atom in modeller.topology.atoms():
        if atom.element.name in ("H", "D"): continue
        p_nm = _atom_pos_nm(modeller.positions, atom.index)
        pos_restraint.addParticle(atom.index, [p_nm[0], p_nm[1], p_nm[2]])
    system.addForce(pos_restraint)

    protein_heavy = [a.index for a in modeller.topology.atoms() if a.element.name not in ("H", "D")]
    insert_force = CustomExternalForce(
        "0.5 * k_insert * select(R0 * scale - sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2 + 0.0001), "
        "(R0 * scale - sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2 + 0.0001))^2, 0)")
    insert_force.addGlobalParameter("k_insert", K_INSERT * unit.kilocalories_per_mole / unit.angstrom ** 2)
    insert_force.addGlobalParameter("scale", 0.0)
    insert_force.addPerParticleParameter("x0"); insert_force.addPerParticleParameter("y0")
    insert_force.addPerParticleParameter("z0"); insert_force.addPerParticleParameter("R0")
    for pos_A, elem in zip(lig_coords_A, lig_elems):
        c_nm = pos_A * 0.1; R0_nm = vdw_of(elem) * 0.1
        for idx in protein_heavy: insert_force.addParticle(idx, [c_nm[0], c_nm[1], c_nm[2], R0_nm])
    system.addForce(insert_force)

    integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP)
    integrator.setRandomNumberSeed(RNG_SEED)
    sim = Simulation(modeller.topology, system, integrator, platform_obj)
    sim.context.setPositions(modeller.positions)

    current_k_pos = k_pos_start
    MAX_SAFE_FORCE = 5000.0
    last_valid_pos = _to_numpy_positions(sim.context.getState(getPositions=True).getPositions())

    print(f"  [自适应松弛] 目标 max|F| < {MAX_SAFE_FORCE:.0f}")
    for relax_round in range(10):
        try:
            for iters, tol in [(500, 1000.0), (1000, 100.0), (2000, 10.0)]:
                sim.minimizeEnergy(maxIterations=iters, tolerance=tol * unit.kilocalories_per_mole / unit.angstrom)
        except Exception as e:
            print(f"    ⚠ Round {relax_round}: minimize 异常: {e}")
            _restore_positions(sim, last_valid_pos)
            current_k_pos *= 0.3
            sim.context.setParameter("k_pos", current_k_pos)
            continue

        mf = _max_force(sim)
        pos_check = _to_numpy_positions(sim.context.getState(getPositions=True).getPositions())

        if not np.all(np.isfinite(pos_check)):
            print(f"    ⚠ Round {relax_round}: NaN，回退并 k_pos *= 0.3")
            _restore_positions(sim, last_valid_pos)
            current_k_pos *= 0.3
            sim.context.setParameter("k_pos", current_k_pos)
            continue

        print(f"    Round {relax_round}: k_pos={current_k_pos:.0f}, max|F|={mf:.0f}")
        if mf <= MAX_SAFE_FORCE:
            print(f"    ✅ 力收敛")
            last_valid_pos = pos_check.copy()
            break

        if current_k_pos > 5.0:
            current_k_pos *= 0.5
            sim.context.setParameter("k_pos", current_k_pos)
            last_valid_pos = pos_check.copy()
        else:
            last_valid_pos = pos_check.copy()
            break

    tracker.record("insert_start", sim.topology, sim.context.getState(getPositions=True).getPositions(), pocket_center_A)

    print(f"  [分级插入]")
    n_broken = 0
    for stage_i, frac in enumerate(SCALE_SCHEDULE, 1):
        k_pos_now = current_k_pos + (K_POS_FINAL - current_k_pos) * (stage_i / N_INSERT_STAGES)
        try:
            sim.context.setParameter("scale", frac)
            sim.context.setParameter("k_pos", k_pos_now)
            sim.minimizeEnergy(maxIterations=MIN_PER_STAGE)
        except Exception as e:
            print(f"    ⛔ 级 {stage_i}: minimize 异常: {e}，回退终止")
            _restore_positions(sim, last_valid_pos); n_broken = stage_i; break

        pos_now = _to_numpy_positions(sim.context.getState(getPositions=True).getPositions())
        mf = _max_force(sim)
        if not np.all(np.isfinite(pos_now)) or mf > 1e7:
            print(f"    ⛔ 级 {stage_i}: NaN 或 力过大({mf:.0f})，回退终止")
            _restore_positions(sim, last_valid_pos); n_broken = stage_i; break

        last_valid_pos = pos_now.copy()
        if stage_i % 3 == 0 or stage_i == N_INSERT_STAGES:
            probe = tracker.record(f"insert_stage_{stage_i}", sim.topology, sim.context.getState(getPositions=True).getPositions(), pocket_center_A)
            print(f"    级 {stage_i}/{N_INSERT_STAGES}: scale={frac:.3f}, probe={probe:+.2f}")

    if n_broken > 0:
        print(f"  ⚠ 插入第 {n_broken} 级中止")

    if not np.all(np.isfinite(_to_numpy_positions(sim.context.getState(getPositions=True).getPositions()))):
        _restore_positions(sim, last_valid_pos)

    safe_write_pdb(sim, out_pdb, fallback_positions_nm=last_valid_pos)
    return out_pdb, n_before, n_after


# ==================== 几何计算辅助 ====================
def _calc_dihedral(p1, p2, p3, p4):
    b1 = p2 - p1; b2 = p3 - p2; b3 = p4 - p3
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))

def _bond_angle_deg(p1, p2, p3):
    v1 = p1 - p2; v2 = p3 - p2
    c = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


# ==================== 主链预过滤（[FIX-1]） ====================
def pre_filter_backbone(pdb_path, rama_thresh=0.70, angle_bad_thresh=50):
    print(f"\n[预过滤] 主链质量: {os.path.basename(pdb_path)}")
    try: st = gemmi.read_structure(pdb_path)
    except Exception as e:
        return False, {"rama_frac": 0.0, "angle_bad": 0, "n_residues_checked": 0, "error": str(e)}

    n_tot, n_ok, n_bad = 0, 0, 0
    for chain in st[0]:
        res = [r for r in chain if r.entity_type == gemmi.EntityType.Polymer]
        for i in range(1, len(res) - 1):
            r0, r1, r2 = res[i-1], res[i], res[i+1]
            try:
                C0=np.array([r0.find_atom("C","*").pos.x,r0.find_atom("C","*").pos.y,r0.find_atom("C","*").pos.z])
                N1=np.array([r1.find_atom("N","*").pos.x,r1.find_atom("N","*").pos.y,r1.find_atom("N","*").pos.z])
                CA1=np.array([r1.find_atom("CA","*").pos.x,r1.find_atom("CA","*").pos.y,r1.find_atom("CA","*").pos.z])
                C1=np.array([r1.find_atom("C","*").pos.x,r1.find_atom("C","*").pos.y,r1.find_atom("C","*").pos.z])
                N2=np.array([r2.find_atom("N","*").pos.x,r2.find_atom("N","*").pos.y,r2.find_atom("N","*").pos.z])
                phi=_calc_dihedral(C0,N1,CA1,C1); psi=_calc_dihedral(N1,CA1,C1,N2); n_tot+=1
                if (-160<phi<-40 and -70<psi<40) or (-160<phi<-50 and 80<psi<180) or (40<phi<90 and -60<psi<180): n_ok+=1
            except: pass
            try:
                N  = np.array([r1.find_atom("N","*").pos.x, r1.find_atom("N","*").pos.y, r1.find_atom("N","*").pos.z])
                CA = np.array([r1.find_atom("CA","*").pos.x, r1.find_atom("CA","*").pos.y, r1.find_atom("CA","*").pos.z])
                C  = np.array([r1.find_atom("C","*").pos.x, r1.find_atom("C","*").pos.y, r1.find_atom("C","*").pos.z])
                if abs(_bond_angle_deg(N, CA, C) - 111.0) > 12: n_bad += 1
                N2 = np.array([r2.find_atom("N","*").pos.x, r2.find_atom("N","*").pos.y, r2.find_atom("N","*").pos.z])
                if abs(_bond_angle_deg(CA, C, N2) - 116.0) > 12: n_bad += 1
            except: pass
    rama_frac = n_ok / n_tot if n_tot else 0.0
    passed = (rama_frac >= rama_thresh) and (n_bad <= angle_bad_thresh)
    status = "✅ 通过" if passed else "❌ 拒绝"
    print(f"  Rama={rama_frac*100:.1f}%, 坏键角={n_bad} → {status}")
    return passed, {"rama_frac": round(rama_frac, 3), "angle_bad": n_bad, "n_residues_checked": n_tot}


# ==================== L1：回弹（[FIX-4]） ====================
def verify_L1_rebound(opened_pdb, pocket_center_A, platform_obj, tracker, case):
    cfg = VERIFY_CONFIG
    print(f"\n[L1 回弹] {cfg['L1_STEPS']} 步")
    pdb = PDBFile(opened_pdb); modeller = Modeller(pdb.topology, pdb.positions)
    ff = ForceField(*FF_FILES)
    system = ff.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=NONBONDED_CUTOFF, constraints=HBonds)
    integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP); integrator.setRandomNumberSeed(RNG_SEED)
    sim = Simulation(modeller.topology, system, integrator, platform_obj)
    sim.context.setPositions(modeller.positions)
    try: sim.minimizeEnergy(maxIterations=2000)
    except Exception as e: print(f"  ⚠ L1 初始最小化失败: {e}")
    start = tracker.record("L1_start", sim.topology, sim.context.getState(getPositions=True).getPositions(), pocket_center_A)
    last_valid = _to_numpy_positions(sim.context.getState(getPositions=True).getPositions())
    try:
        for ci in range(0, cfg["L1_STEPS"], cfg["L1_MONITOR"]):
            sim.step(cfg["L1_MONITOR"])
            cur = _to_numpy_positions(sim.context.getState(getPositions=True).getPositions())
            if np.all(np.isfinite(cur)): last_valid = cur
            # [FIX-4] 每个 monitor 都记录
            tracker.record(f"L1_{ci+cfg['L1_MONITOR']}", sim.topology, sim.context.getState(getPositions=True).getPositions(), pocket_center_A)
    except Exception as e:
        print(f"  ⚠ L1 MD 异常: {e}"); _restore_positions(sim, last_valid)
    end = tracker.history[-1][1]; drop = start - end
    verdict = "不可维持" if drop > 0.5 else ("亚稳态" if drop > 0.2 else "可维持")
    print(f"  L1 回弹量 {drop:+.2f} Å → {verdict}")
    safe_write_pdb(sim, os.path.join(OUT_DIR, f"{case}_L1_rebound.pdb"), fallback_positions_nm=last_valid)
    return {"L1_start": round(start,3), "L1_end": round(end,3), "L1_drop": round(drop,3), "L1_verdict": verdict}


# ==================== L2 ====================
def verify_L2_sanity(pdb_path, ref_pdb_path, case):
    print(f"\n[L2 结构合理性]")
    st = gemmi.read_structure(pdb_path)
    n_tot, n_ok = 0, 0
    for chain in st[0]:
        res = [r for r in chain if r.entity_type == gemmi.EntityType.Polymer]
        for i in range(1, len(res)-1):
            r0, r1, r2 = res[i-1], res[i], res[i+1]
            try:
                C0=np.array([r0.find_atom("C","*").pos.x,r0.find_atom("C","*").pos.y,r0.find_atom("C","*").pos.z])
                N1=np.array([r1.find_atom("N","*").pos.x,r1.find_atom("N","*").pos.y,r1.find_atom("N","*").pos.z])
                CA1=np.array([r1.find_atom("CA","*").pos.x,r1.find_atom("CA","*").pos.y,r1.find_atom("CA","*").pos.z])
                C1=np.array([r1.find_atom("C","*").pos.x,r1.find_atom("C","*").pos.y,r1.find_atom("C","*").pos.z])
                N2=np.array([r2.find_atom("N","*").pos.x,r2.find_atom("N","*").pos.y,r2.find_atom("N","*").pos.z])
                phi=_calc_dihedral(C0,N1,CA1,C1); psi=_calc_dihedral(N1,CA1,C1,N2); n_tot+=1
                if (-160<phi<-40 and -70<psi<40) or (-160<phi<-50 and 80<psi<180) or (40<phi<90 and -60<psi<180): n_ok+=1
            except: pass
    rama_frac = n_ok / n_tot if n_tot else 0.0
    n_clash = count_clashes_st(st)
    n_bad_ang, max_dev_ang = 0, 0.0
    for chain in st[0]:
        res_list = [r for r in chain if r.entity_type == gemmi.EntityType.Polymer]
        for i, r in enumerate(res_list):
            try:
                N  = np.array([r.find_atom("N","*").pos.x, r.find_atom("N","*").pos.y, r.find_atom("N","*").pos.z])
                CA = np.array([r.find_atom("CA","*").pos.x, r.find_atom("CA","*").pos.y, r.find_atom("CA","*").pos.z])
                C  = np.array([r.find_atom("C","*").pos.x, r.find_atom("C","*").pos.y, r.find_atom("C","*").pos.z])
                dev = abs(_bond_angle_deg(N, CA, C) - 111.0)
                if dev > 12: n_bad_ang += 1
                max_dev_ang = max(max_dev_ang, dev)
            except: pass
    rmsd = ca_rmsd_after_align(st, gemmi.read_structure(ref_pdb_path)) if ref_pdb_path and os.path.exists(ref_pdb_path) else None
    issues = []
    if rama_frac < 0.90: issues.append(f"Rama {rama_frac*100:.1f}%")
    if n_clash > 10: issues.append(f"clash={n_clash}")
    if n_bad_ang > 20: issues.append(f"键角={n_bad_ang}")
    verdict = ("异常: " + ", ".join(issues)) if issues else "合理"
    print(f"  Rama={rama_frac*100:.1f}%, Clash={n_clash}, 坏键角={n_bad_ang}")
    if rmsd: print(f"  CA-RMSD: {rmsd:.2f} Å")
    print(f"  L2: {verdict}")
    return {"L2_rama": round(rama_frac, 3), "L2_clash": n_clash, "L2_rmsd": round(rmsd,2) if rmsd else "", "L2_verdict": verdict}


# ==================== L3：直方图 PMF（[FIX-3]） ====================
class HistogramPMF:
    def __init__(self, s_min, s_max, n_bins):
        self.edges = np.linspace(s_min, s_max, n_bins + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.n_bins = n_bins
        self.count = np.zeros(n_bins, dtype=int)
    def bin_of(self, s):
        return int(np.clip(np.searchsorted(self.edges, s) - 1, 0, self.n_bins - 1))
    def accumulate(self, s): self.count[self.bin_of(s)] += 1
    def calc_pmf_kJ(self, temp_K=300.0):
        kT = 8.314 * temp_K / 1000.0
        hist = self.count.astype(float)
        hist[hist == 0] = np.nan
        pmf = -kT * np.log(hist / np.nanmax(hist))
        pmf -= np.nanmin(pmf)
        return np.where(np.isnan(pmf), 0, pmf)

def build_soft_count_cv(topology, pocket_center_A, R0_A, sigma_A):
    inner = CustomExternalForce(
        "1 / (1 + exp((sqrt((x - cx_cv)^2 + (y - cy_cv)^2 + (z - cz_cv)^2 + 1e-8) - r0_cv) / sigma_cv))")
    inner.addGlobalParameter("r0_cv", R0_A * 0.1)
    inner.addGlobalParameter("sigma_cv", sigma_A * 0.1)
    c = np.asarray(pocket_center_A, dtype=float) * 0.1
    inner.addGlobalParameter("cx_cv", c[0]); inner.addGlobalParameter("cy_cv", c[1]); inner.addGlobalParameter("cz_cv", c[2])
    heavy_idx = []
    for atom in topology.atoms():
        if atom.element.name in ("H","D"): continue
        inner.addParticle(atom.index); heavy_idx.append(atom.index)
    cv = CustomCVForce("s_val")
    cv.addCollectiveVariable("s_val", inner)
    return cv, heavy_idx

def _probe_s_range(pdb_path, pocket_center_A, platform_obj):
    cfg = VERIFY_CONFIG
    pdb = PDBFile(pdb_path); modeller = Modeller(pdb.topology, pdb.positions)
    ff = ForceField(*FF_FILES)
    system = ff.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=NONBONDED_CUTOFF, constraints=HBonds)
    cv_force, _ = build_soft_count_cv(modeller.topology, pocket_center_A, cfg["L3_CV_R0"], cfg["L3_CV_SIGMA"])
    system.addForce(cv_force)
    integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP); integrator.setRandomNumberSeed(RNG_SEED)
    sim = Simulation(modeller.topology, system, integrator, platform_obj)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=1000)
    sim.step(cfg["L3_PROBE_STEPS"])
    s_vals = cv_force.getCollectiveVariableValues(sim.context)
    return float(s_vals[0]) if hasattr(s_vals, "__len__") else float(s_vals)

def verify_L3_pmf(apo_pdb, opened_pdb, pocket_center_A, platform_obj, case):
    cfg = VERIFY_CONFIG
    print(f"\n[L3 无偏采样 PMF]")

    # [FIX-3] 自适应探测 s 范围
    s_open = _probe_s_range(opened_pdb, pocket_center_A, platform_obj)
    s_apo = _probe_s_range(apo_pdb, pocket_center_A, platform_obj)
    margin = cfg["L3_RANGE_MARGIN"]
    s_range = (max(0.0, min(s_open, s_apo) - margin), max(s_open, s_apo) + margin)
    print(f"  s(opened)={s_open:.2f}, s(apo)={s_apo:.2f}, 范围={s_range}")

    pdb = PDBFile(opened_pdb); modeller = Modeller(pdb.topology, pdb.positions)
    ff = ForceField(*FF_FILES)
    system = ff.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=NONBONDED_CUTOFF, constraints=HBonds)
    cv_force, _ = build_soft_count_cv(modeller.topology, pocket_center_A, cfg["L3_CV_R0"], cfg["L3_CV_SIGMA"])
    system.addForce(cv_force)
    integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP); integrator.setRandomNumberSeed(RNG_SEED)
    sim = Simulation(modeller.topology, system, integrator, platform_obj)
    sim.context.setPositions(modeller.positions)
    try: sim.minimizeEnergy(maxIterations=2000)
    except Exception as e: print(f"  ⚠ L3 初始最小化失败: {e}")
    pmf_calc = HistogramPMF(s_range[0], s_range[1], cfg["L3_N_BINS"])
    last_valid = _to_numpy_positions(sim.context.getState(getPositions=True).getPositions())
    for step in range(0, cfg["L3_TOTAL_STEPS"], cfg["L3_UPDATE_EVERY"]):
        try: sim.step(cfg["L3_UPDATE_EVERY"])
        except Exception as e:
            print(f"  ⚠ L3 MD 失败（{e}），提前结束")
            _restore_positions(sim, last_valid); break
        cur = _to_numpy_positions(sim.context.getState(getPositions=True).getPositions())
        if np.all(np.isfinite(cur)): last_valid = cur
        s_now = float(cv_force.getCollectiveVariableValues(sim.context)[0])
        pmf_calc.accumulate(s_now)
        if (step // cfg["L3_UPDATE_EVERY"]) % 50 == 0:
            print(f"    步 {step}: s={s_now:.2f}")
    fs_csv = os.path.join(OUT_DIR, f"{case}_L3_F_s.csv")
    with open(fs_csv, "w", newline="", encoding="utf-8-sig") as f:
        f.write("s_center,PMF_kJ_mol,samples\n")
        pmf_vals = pmf_calc.calc_pmf_kJ()
        for i in range(pmf_calc.n_bins): f.write(f"{pmf_calc.centers[i]:.3f},{pmf_vals[i]:.3f},{pmf_calc.count[i]}\n")
    F_range = float(np.max(pmf_vals))
    verdict = "热力学不可行 (>30 kJ/mol)" if F_range > 30 else ("亚稳态 (10-30 kJ/mol)" if F_range > 10 else "热力学可打开 (<10 kJ/mol)")
    print(f"  L3 PMF 范围: {F_range:.1f} kJ/mol → {verdict}")
    return {"L3_F_range_kJ": round(F_range,1), "L3_verdict": verdict}


# ==================== L4-lite ====================
def verify_L4_lite(opened_pdb, lig_coords_A, lig_elems, pocket_center_A, platform_obj, tracker, case):
    cfg = VERIFY_CONFIG
    print(f"\n[L4-lite] {cfg['L4_STEPS']} 步")
    pdb = PDBFile(opened_pdb); modeller = Modeller(pdb.topology, pdb.positions)
    ff = ForceField(*FF_FILES)
    system = ff.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=NONBONDED_CUTOFF, constraints=HBonds)
    lock = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    lock.addPerParticleParameter("k"); lock.addPerParticleParameter("x0"); lock.addPerParticleParameter("y0"); lock.addPerParticleParameter("z0")
    for atom in modeller.topology.atoms():
        if atom.element.name in ("H", "D") or atom.name not in BACKBONE_NAMES: continue
        p_nm = _atom_pos_nm(modeller.positions, atom.index)
        lock.addParticle(atom.index, [k_quantity(cfg["L4_K_BACKBONE"]), p_nm[0], p_nm[1], p_nm[2]])
    system.addForce(lock)
    protein_heavy = [a.index for a in modeller.topology.atoms() if a.element.name not in ("H", "D")]
    insert_force = CustomExternalForce(
        "0.5 * k_insert * select(R0 * scale - sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2 + 0.0001), "
        "(R0 * scale - sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2 + 0.0001))^2, 0)")
    insert_force.addGlobalParameter("k_insert", K_INSERT * unit.kilocalories_per_mole / unit.angstrom ** 2)
    insert_force.addGlobalParameter("scale", 1.0)
    insert_force.addPerParticleParameter("x0"); insert_force.addPerParticleParameter("y0")
    insert_force.addPerParticleParameter("z0"); insert_force.addPerParticleParameter("R0")
    for pos_A, elem in zip(lig_coords_A, lig_elems):
        c_nm = pos_A * 0.1; R0_nm = vdw_of(elem) * 0.1
        for idx in protein_heavy: insert_force.addParticle(idx, [c_nm[0], c_nm[1], c_nm[2], R0_nm])
    system.addForce(insert_force)
    integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP); integrator.setRandomNumberSeed(RNG_SEED)
    sim = Simulation(modeller.topology, system, integrator, platform_obj)
    sim.context.setPositions(modeller.positions)
    try: sim.minimizeEnergy(maxIterations=2000)
    except Exception as e: print(f"  ⚠ L4 初始最小化失败: {e}")
    start = tracker.record("L4_start", sim.topology, sim.context.getState(getPositions=True).getPositions(), pocket_center_A)
    last_valid = _to_numpy_positions(sim.context.getState(getPositions=True).getPositions())
    try:
        for ci in range(0, cfg["L4_STEPS"], cfg["L4_MONITOR"]):
            sim.step(cfg["L4_MONITOR"])
            cur = _to_numpy_positions(sim.context.getState(getPositions=True).getPositions())
            if np.all(np.isfinite(cur)): last_valid = cur
    except Exception as e:
        print(f"  ⚠ L4 MD 失败: {e}"); _restore_positions(sim, last_valid)
    end = tracker.record("L4_end", sim.topology, sim.context.getState(getPositions=True).getPositions(), pocket_center_A)
    drop = start - end
    verdict = "排斥力也维持不住" if drop > 0.5 else ("排斥力勉强维持" if drop > 0.2 else "排斥力可维持（不等价于 holo 稳定性）")
    print(f"  L4: {start:.2f} → {end:.2f} (Δ{drop:+.2f}) → {verdict}")
    safe_write_pdb(sim, os.path.join(OUT_DIR, f"{case}_L4_final.pdb"), fallback_positions_nm=last_valid)
    return {"L4_start": round(start,3), "L4_end": round(end,3), "L4_drop": round(drop,3), "L4_verdict": verdict}


# ==================== CSV 追加 ====================
def _append_result(csv_path, result_dict):
    file_exists = os.path.exists(csv_path)
    write_header = not file_exists or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_KEYS, extrasaction='ignore')
        if write_header: w.writeheader()
        w.writerow({k: result_dict.get(k, "") for k in CSV_KEYS})


# ==================== 主流程 ====================
def process_case(case):
    print(f"\n{'='*70}\n处理 {case}\n{'='*70}")
    cif, seed = find_best_seed(case)
    if cif is None: return {"case": case, "error": "find_best_seed None"}
    platform_obj = pick_platform()

    st = gemmi.read_structure(cif)
    apo_raw = os.path.join(OUT_DIR, f"{case}_apo_raw.pdb")
    strip_nonpolymer(st).write_pdb(apo_raw)

    apo_pdb = os.path.join(OUT_DIR, f"{case}_apo.pdb")
    if not fix_pdb_with_fixer(apo_raw, apo_pdb): return {"case": case, "error": "PDBFixer 失败"}

    bb_passed, bb_metrics = pre_filter_backbone(apo_pdb)
    if not bb_passed:
        print(f"⛔ {case} 预过滤拒绝")
        return {
            "case": case, "seed": seed["dir"], "seed_probe": round(seed["probe"],3),
            "seed_clash": seed["clash"], "seed_init_clash": seed["init_clash"],
            "pre_filtered": True,
            "filter_rama": bb_metrics.get("rama_frac", 0.0),
            "filter_angle_bad": bb_metrics.get("angle_bad", 0),
            "L1_verdict": "跳过(主链异常)", "L2_verdict": "跳过(主链异常)",
            "L3_verdict": "跳过(主链异常)", "L4_verdict": "跳过(主链异常)"
        }

    tracker = ProbeTracker()
    pdb0 = PDBFile(apo_pdb)
    tracker.record("seed", pdb0.topology, pdb0.positions, seed["tc_af3"])

    work_pdb = apo_pdb
    preheated_flag = 0
    if ENABLE_PREHEAT and seed["init_clash"] > PREHEAT_CLASH_THRESH:
        print(f"\n[触发平铺] init_clash={seed['init_clash']}")
        preheated = os.path.join(OUT_DIR, f"{case}_preheated.pdb")
        preheat_unfold(apo_pdb, seed["tc_af3"], platform_obj, preheated)
        work_pdb = preheated
        preheated_flag = 1
        p_pre = PDBFile(preheated)
        tracker.record("preheated", p_pre.topology, p_pre.positions, seed["tc_af3"])

    opened_pdb = os.path.join(OUT_DIR, f"{case}_opened.pdb")
    opened_result = steered_insertion(work_pdb, seed["lig_af3"], seed["lig_elems"], seed["tc_af3"],
                                       platform_obj, tracker, opened_pdb, seed_info=seed)

    if opened_result is None:
        return {"case": case, "seed": seed["dir"], "error": "初始几何冲突过多",
                "seed_probe": round(seed["probe"],3), "seed_clash": seed["clash"],
                "seed_init_clash": seed["init_clash"], "geom_rejected": True}

    _, n_before, n_after = opened_result

    results = {"case": case, "seed": seed["dir"], "seed_probe": round(seed["probe"],3),
               "seed_clash": seed["clash"], "seed_init_clash": seed["init_clash"],
               "pre_filtered": False, "preheated": preheated_flag,
               "geom_clash_before": n_before, "geom_clash_after": n_after,
               "geom_rejected": False}

    for key, fn in [
        ("L1", lambda: verify_L1_rebound(opened_pdb, seed["tc_af3"], platform_obj, tracker, case)),
        ("L2", lambda: verify_L2_sanity(opened_pdb, os.path.join(EXP_DIR, case, "receptor.pdb"), case)),
        ("L3", lambda: verify_L3_pmf(apo_pdb, opened_pdb, seed["tc_af3"], platform_obj, case)),
        ("L4", lambda: verify_L4_lite(opened_pdb, seed["lig_af3"], seed["lig_elems"], seed["tc_af3"], platform_obj, tracker, case)),
    ]:
        flag = f"RUN_{key}" if key in ("L1", "L2", "L3") else f"RUN_{key}_LITE"
        if not VERIFY_CONFIG.get(flag, True): continue
        try: results.update(fn())
        except Exception as e:
            print(f"  [{key} 失败] {e}"); traceback.print_exc()

    tracker.save_csv(os.path.join(OUT_DIR, f"{case}_probe_trace.csv"))

    print(f"\n{'='*70}\n{case} 判决\n{'='*70}")
    for k in ("L1_verdict","L2_verdict","L3_verdict","L4_verdict"):
        print(f"  {k}: {results.get(k, 'N/A')}")

    return results


if __name__ == "__main__":
    csv_path = os.path.join(OUT_DIR, "verification_summary.csv")
    if os.path.exists(csv_path):
        backup = csv_path + ".v12_bak"
        if not os.path.exists(backup):
            shutil.copy2(csv_path, backup)
            print(f"[备份] 旧 CSV → {backup}")

    print(f"\n{'='*60}")
    print(f"📋 跑指定的 {len(TARGET_CASES)} 个 case（v13.0 修复版）")
    print(f"{'='*60}\n")

    for idx, case in enumerate(TARGET_CASES, 1):
        print(f"\n[{idx}/{len(TARGET_CASES)}] ▶ {case}")
        try:
            result = process_case(case)
            if result is not None:
                _append_result(csv_path, result)
                print(f"[{idx}/{len(TARGET_CASES)}] ✅ {case}")
        except KeyboardInterrupt:
            print(f"\n⛔ Ctrl+C 中断"); sys.exit(0)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:200]}"
            try: _append_result(csv_path, {"case": case, "error": err_msg})
            except: pass
            print(f"[{idx}/{len(TARGET_CASES)}] ❌ {case}: {err_msg}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"🏁 全部完成。结果 → {csv_path}")
    print(f"{'='*60}")