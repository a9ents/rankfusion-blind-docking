#!/usr/bin/env python3
"""AF3 输出解析 / 叠合 / 几何计算的公共工具。"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

# ============ 配置 ============
# 假设 USalign 与项目根目录的相对位置不变，若实际路径不同，请修改此处
# 项目根目录：<repo_root>/src/af3_common.py -> 上两级是 repo_root
_REPO_ROOT = Path(__file__).resolve().parents[2]
USALIGN_EXE = str(_REPO_ROOT / "USalign" / "USalign.exe")   # 若实际位置不同，改这里


# ============ 结构解析 ============
def split_af3_cif(cif_path, out_receptor_pdb, out_ligand_pdb=None):
    """
    把 AF3 mmCIF 拆成受体 PDB 和配体 PDB（共折叠时配体存在）。
    返回 (receptor_ok, ligand_ok)。
    """
    try:
        import gemmi
    except ImportError:
        print("    ⚠ 需要 gemmi: pip install gemmi")
        return False, False

    try:
        st = gemmi.read_structure(str(cif_path))
        st.setup_entities()
        st.remove_alternative_conformations()
        st.remove_hydrogens()
        st.remove_waters()
    except Exception as e:
        print(f"    ⚠ 读取 CIF 失败: {e}")
        return False, False

    prot_chains, lig_chains = [], []
    for chain in st[0]:
        aa3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
               "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
        nt3 = {"A","C","G","U","DA","DC","DG","DT"}
        is_prot = any(
            (r.name.strip().upper() in aa3) or (r.name.strip().upper() in nt3)
            for r in chain
        )
        (prot_chains if is_prot else lig_chains).append(chain.name)

    receptor_ok = False
    if prot_chains:
        try:
            st_prot = st.clone()
            for ch in lig_chains:
                st_prot[0].remove_chain(ch)
            st_prot.write_pdb(str(out_receptor_pdb))
            receptor_ok = True
        except Exception as e:
            print(f"    ⚠ 写受体失败: {e}")

    ligand_ok = False
    if out_ligand_pdb is not None and lig_chains:
        try:
            st_lig = st.clone()
            for ch in prot_chains:
                st_lig[0].remove_chain(ch)
            st_lig.write_pdb(str(out_ligand_pdb))
            ligand_ok = True
        except Exception as e:
            print(f"    ⚠ 写配体失败: {e}")

    return receptor_ok, ligand_ok


def load_af3_summary_confidence(cif_path):
    """读取 CIF 同目录的 summary confidences json（兼容 AF3 各种命名）。"""
    d = Path(cif_path).parent

    candidates = []
    for pat in ("*SUMMARY_CONFIDENCES*.JSON",
                "*SUMMARY_CONFIDENCES*.json",
                "*summary_confidences*.json",
                "*summary_confidences*.JSON"):
        candidates.extend(d.glob(pat))

    if not candidates:
        for pat in ("*CONFIDENCES*.JSON", "*confidences*.json"):
            candidates.extend(d.glob(pat))

    for p in candidates:
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if "ranking_score" not in data:
            for k in ("overall_confidence", "ranking_confidence",
                      "iptm", "ptm"):
                if k in data:
                    data["ranking_score"] = data.get(k, -1.0)
                    break

        if "iptm" not in data and "ptm" in data:
            data["iptm"] = data["ptm"]

        return data
    return {}


# ============ 叠合 ============
def run_usalign(mobile_pdb, target_pdb, out_pdb, timeout=120):
    """US-align 叠合，返回 (RMSD, TM-score)。"""
    out_pdb = Path(out_pdb).resolve()
    prefix = str(out_pdb.with_suffix(""))
    cmd = [USALIGN_EXE,
           str(Path(mobile_pdb).resolve()),
           str(Path(target_pdb).resolve()),
           "-mm", "1", "-o", prefix]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        print(f"    ⚠ 找不到 USalign: {USALIGN_EXE}")
        return None, None
    except subprocess.TimeoutExpired:
        return None, None

    rmsd = tm = None
    for line in r.stdout.splitlines():
        if rmsd is None and "RMSD=" in line:
            try:
                rmsd = float(line.split("RMSD=")[1].split(",")[0].strip())
            except (IndexError, ValueError):
                pass
        if tm is None and "TM-score=" in line:
            try:
                tm = float(line.split("TM-score=")[1].split()[0].strip())
            except (IndexError, ValueError):
                pass

    actual = Path(prefix + ".pdb")
    if actual.is_file() and actual != out_pdb:
        shutil.move(str(actual), str(out_pdb))
    for f in actual.parent.glob(actual.stem + "*.pml"):
        try:
            f.unlink()
        except OSError:
            pass
    return rmsd, tm


# ============ 几何 ============
def ligand_centroid(ligand_path):
    """重原子质心。支持 .sdf/.mol/.pdb/.mol2。"""
    p = Path(ligand_path)
    suffix = p.suffix.lower()

    if suffix in (".sdf", ".mol"):
        try:
            from rdkit import Chem
            mol = Chem.MolFromMolFile(str(p), removeHs=True)
            if mol is None:
                return None
            conf = mol.GetConformer()
            coords = []
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() > 1:
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coords.append([pos.x, pos.y, pos.z])
            return np.mean(coords, axis=0) if coords else None
        except Exception:
            return None

    if suffix in (".pdb", ".ent"):
        coords = []
        with open(p) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    elem = line[76:78].strip().upper() if len(line) > 76 else ""
                    if elem in ("H", "D"):
                        continue
                    try:
                        coords.append([float(line[30:38]),
                                       float(line[38:46]),
                                       float(line[46:54])])
                    except ValueError:
                        continue
        return np.mean(coords, axis=0) if coords else None

    if suffix == ".mol2":
        coords = []
        with open(p) as f:
            in_atom = False
            for line in f:
                if line.startswith("@<TRIPOS>ATOM"):
                    in_atom = True
                    continue
                if line.startswith("@<TRIPOS>") and in_atom:
                    break
                if in_atom:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            coords.append([float(parts[2]),
                                           float(parts[3]),
                                           float(parts[4])])
                        except ValueError:
                            continue
        return np.mean(coords, axis=0) if coords else None

    return None


def parse_box_center(box_path):
    """解析 Vina .box 中心。支持 center_x= 和裸数字两种格式。"""
    p = Path(box_path)
    if not p.is_file():
        return None
    center = {}
    plain = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip().lower()
                try:
                    v = float(v.strip())
                except ValueError:
                    continue
                if k in ("center_x", "center_y", "center_z"):
                    center[k] = v
            else:
                parts = line.replace(",", " ").split()
                try:
                    nums = [float(x) for x in parts]
                    if len(nums) >= 3:
                        plain.append(nums[:3])
                except ValueError:
                    pass
    if len(center) == 3:
        return np.array([center["center_x"],
                         center["center_y"],
                         center["center_z"]])
    return np.array(plain[0]) if plain else None


def euclidean(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))