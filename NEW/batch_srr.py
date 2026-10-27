import os
import glob
import csv
import json
import numpy as np
import gemmi
import difflib
from pathlib import Path
from openmm.app import PDBFile, Modeller, ForceField, Simulation, CutoffNonPeriodic, HBonds
from openmm import LangevinMiddleIntegrator, Platform, unit, CustomExternalForce
from pdbfixer import PDBFixer
from dlpacker import DLPacker

# ================= 路径配置 =================
CIF_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3_workflow"
APO_DIR = r"E:\AF3环境\ALL-DATA\DATA\AF3-PBD"
EXP_DIR = r"E:\AF3环境\ALL-DATA\DATA\exp_structures"
LIG_DIR = r"E:\AF3环境\ALL-DATA\DATA\ligands"
OUT_DIR = r"E:\AF3环境\ALL-DATA\SRR_RESULTS"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUT_DIR, "srr_low_conf_results.csv") 
FINAL_COMPARISON_CSV = r"E:\AF3环境\ALL-DATA\probe_final_comparison\final_comparison.csv"

VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "H": 1.20}
def vdw_of(elem): return VDW.get(elem.upper(), 1.70)

def calc_probe(pdb_path, center):
    st = gemmi.read_structure(str(pdb_path))
    st.remove_alternative_conformations(); st.remove_hydrogens(); st.remove_waters()
    c = np.array(center, dtype=float)
    min_clearance = 5.0
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element.name in ("H", "D"): continue
                p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                d = np.linalg.norm(p - c)
                if d < 10.0:
                    clearance = d - vdw_of(atom.element.name)
                    if clearance < min_clearance: min_clearance = clearance
    return max(min_clearance, 0.0)

def get_ca_list(pdb):
    st = gemmi.read_structure(str(pdb))
    out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.name == "CA":
                    out.append((chain.name, res.seqid.num, res.name, np.array([atom.pos.x, atom.pos.y, atom.pos.z])))
                    break
    return out

def match_seq(a, b):
    sa = [x[2] for x in a]; sb = [x[2] for x in b]
    sm = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
    return [(blk.a + k, blk.b + k) for blk in sm.get_matching_blocks() for k in range(blk.size)]

def kabsch(P, Q):
    Pc = P.mean(0); Qc = Q.mean(0); P0 = P - Pc; Q0 = Q - Qc
    H = P0.T @ Q0; U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc

def get_true_center(lig_path):
    coords = []
    p = Path(lig_path); suf = p.suffix.lower()
    if suf in (".sdf", ".mol"):
        with open(p, encoding="utf-8", errors="ignore") as f: lines = f.readlines()
        n = int(lines[3][0:3].strip())
        for i in range(4, 4 + n): coords.append([float(lines[i][0:10]), float(lines[i][10:20]), float(lines[i][20:30])])
    elif suf == ".pdb":
        with open(p) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    elem = line[76:78].strip().upper() if len(line) > 76 else ""
                    if elem in ("H", "D"): continue
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.mean(coords, axis=0) if coords else None

def process_case(case_name):
    apo_pdb = os.path.join(APO_DIR, f"{case_name}_apo.pdb")
    exp_rec = os.path.join(EXP_DIR, case_name, "receptor.pdb")
    
    lig_file = None
    for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb"):
        p = os.path.join(LIG_DIR, case_name, nm)
        if os.path.exists(p): lig_file = p; break

    if not (os.path.exists(apo_pdb) and os.path.exists(exp_rec) and lig_file):
        return None

    try:
        tc = get_true_center(lig_file)
        ca_af3 = get_ca_list(apo_pdb); ca_exp = get_ca_list(exp_rec)
        pairs = match_seq(ca_af3, ca_exp)
        if len(pairs) < 10: return None
        P = np.array([ca_af3[i][3] for i, _ in pairs]); Q = np.array([ca_exp[j][3] for _, j in pairs])
        R, t = kabsch(P, Q)
        tc_af3 = R.T @ tc - (R.T @ t)
        r_orig = calc_probe(apo_pdb, tc_af3)
    except Exception:
        return None

    poly_pdb = os.path.join(OUT_DIR, f"{case_name}_poly.pdb")
    relax_pdb = os.path.join(OUT_DIR, f"{case_name}_relax.pdb")
    final_pdb = os.path.join(OUT_DIR, f"{case_name}_final.pdb")

    try:
        # A. 剥离
        st = gemmi.read_structure(apo_pdb)
        for model in st:
            for chain in model:
                for res in chain:
                    for i in range(len(res) - 1, -1, -1):
                        atom = res[i]
                        if atom.name not in ["N", "CA", "C", "O", "OXT"]:
                            try: res.remove_atom(atom.name, atom.altloc if atom.altloc else "")
                            except RuntimeError: pass
                    res.name = "GLY"
        st.write_pdb(poly_pdb)

        # B. 软化
        fixer = PDBFixer(filename=poly_pdb)
        fixer.findMissingResidues(); fixer.findMissingAtoms(); fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)
        with open(poly_pdb, 'w') as f: PDBFile.writeFile(fixer.topology, fixer.positions, f)

        pdb = PDBFile(poly_pdb)
        modeller = Modeller(pdb.topology, pdb.positions)
        forcefield = ForceField('amber14-all.xml', 'implicit/obc2.xml')
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometer, constraints=HBonds)

        restraint = CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        restraint.addGlobalParameter('k', 50.0 * unit.kilocalories_per_mole / unit.angstroms**2)
        restraint.addPerParticleParameter('x0'); restraint.addPerParticleParameter('y0'); restraint.addPerParticleParameter('z0')

        bubble = CustomExternalForce("step(R - r) * kbubble * (R - r)^2; r = sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2)")
        bubble.addGlobalParameter('R', 2.0 * unit.angstroms)
        bubble.addGlobalParameter('kbubble', 50.0 * unit.kilocalories_per_mole / unit.angstroms**2)
        bubble.addGlobalParameter('cx', tc_af3[0] * unit.angstroms)
        bubble.addGlobalParameter('cy', tc_af3[1] * unit.angstroms)
        bubble.addGlobalParameter('cz', tc_af3[2] * unit.angstroms)

        for atom in modeller.topology.atoms():
            if atom.name in ['CA', 'C', 'N', 'O']:
                pos = modeller.positions[atom.index]
                dist = np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - tc_af3)
                if dist > 12.0: restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
                else: bubble.addParticle(atom.index)

        system.addForce(restraint); system.addForce(bubble)
        integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        
        try: platform = Platform.getPlatformByName('CUDA')
        except Exception: platform = Platform.getPlatformByName('CPU')
            
        simulation = Simulation(modeller.topology, system, integrator, platform)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(maxIterations=2000)

        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(relax_pdb, 'w') as f: PDBFile.writeFile(simulation.topology, positions, f)
        r_relax = calc_probe(relax_pdb, tc_af3)

        # C. 重装
        dlp = DLPacker(relax_pdb)
        dlp.reconstruct_protein(order='sequence', output_filename=final_pdb)
        r_final = calc_probe(final_pdb, tc_af3)

        return {"case": case_name, "probe_af3": round(r_orig, 2), "probe_relax": round(r_relax, 2), "probe_final": round(r_final, 2)}
    except Exception as e:
        print(f"  ❌ {case_name} 报错: {e}")
        return {"case": case_name, "probe_af3": round(r_orig, 2), "probe_relax": "error", "probe_final": "error"}

if __name__ == "__main__":
    print("正在读取比对数据进行条件筛选 (ipTM < 0.6, 探针 < 0.5)...")
    
    # 🌟 这里去掉了 pandas，直接用 csv 模块读取
    cases = []
    with open(FINAL_COMPARISON_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                p_af3 = float(row['probe_af3'])
                p_omm = float(row['probe_openmm'])
                p_dlp = float(row['probe_dlp'])
            except (ValueError, KeyError):
                continue
                
            if p_af3 < 0.5 and p_omm < 0.5 and p_dlp < 0.5:
                case = row['case']
                json_path = os.path.join(CIF_DIR, case, f"{case}_summary_confidences.json")
                json_path_alt = os.path.join(CIF_DIR, case, f"{case}_CONFIDENCES.JSON")
                iptm = None
                for jp in [json_path, json_path_alt]:
                    if os.path.exists(jp):
                        with open(jp) as fjson: 
                            iptm = json.load(fjson).get('iptm')
                        break
                if iptm is not None and iptm < 0.6:
                    cases.append(case)
            
    print(f"🎯 筛选完毕！锁定 {len(cases)} 个 ipTM < 0.6 的死结样本。")
    
    # 读取已有进度
    processed = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader: processed.add(row["case"])
            
    print(f"待处理: {len(cases) - len(processed)} 个。")
    
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["case", "probe_af3", "probe_relax", "probe_final"])

    for idx, case in enumerate(cases, 1):
        if case in processed: continue
        print(f"[{idx}/{len(cases)}] 正在强攻样本: {case} ...")
        
        res = process_case(case)
        if res:
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["case", "probe_af3", "probe_relax", "probe_final"])
                w.writerow(res)
                
    print(f"\n🎉 批量 SRR 处理完成！结果已保存至 {CSV_PATH}")