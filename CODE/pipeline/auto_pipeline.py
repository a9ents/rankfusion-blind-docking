#!/usr/bin/env python3
"""
auto_pipeline.py — 全自动对接流程
  - 单盒模式 (single)：bbox + max_size 截断
  - 网格模式 (grid)：切网格 + 过滤空盒子
  - ★ 口袋引导模式 (pocket_guided)：fpocket + Rank Fusion 选盒子
  - 系综对接、共识盒子、共识打分可选
  - 自动遍历大分子目录

依赖：vina, obabel, meeko
      pocket_guided 模式需要：WSL + fpocket (conda install -c conda-forge fpocket)
"""

import os
import sys
import csv
import json          # >>> AF3 DUMP 需要
import shutil
import subprocess
import tempfile
import multiprocessing
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.cluster import DBSCAN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# 工具探测
# =====================================================================
def _which(name):
    p = shutil.which(name)
    if p:
        return p
    for ext in ("", ".exe", ".bat", ".cmd", ".py"):
        cand = os.path.join(SCRIPT_DIR, name + ext)
        if os.path.isfile(cand):
            return cand
    py_dir = os.path.dirname(sys.executable)
    for sd in (os.path.join(py_dir, "Scripts"), py_dir):
        for ext in ("", ".exe", ".bat", ".cmd", ".py"):
            cand = os.path.join(sd, name + ext)
            if os.path.isfile(cand):
                return cand
    return None


VINA_BIN = _which("vina") or r"E:\tools\vina\vina.exe"
OBABEL_BIN = _which("obabel")
MEEKO_BIN = _which("mk_prepare_ligand")

_VINA_GPU_MANUAL = r"E:\AUTO\Vina-GPU-2.1-main (1)\Vina-GPU-2.1-main\AutoDock-Vina-GPU-2.1\Vina-GPU+.exe"
VINA_GPU_BIN = None
if _VINA_GPU_MANUAL and os.path.isfile(_VINA_GPU_MANUAL):
    VINA_GPU_BIN = _VINA_GPU_MANUAL
else:
    VINA_GPU_BIN = (_which("AutoDock-Vina-GPU-2-1") or _which("Vina-GPU+"))

_TOOL_CACHE = {}


def tool_available(name):
    if name in _TOOL_CACHE:
        return _TOOL_CACHE[name]
    ok = (shutil.which(name) is not None) or os.path.isfile(name)
    _TOOL_CACHE[name] = ok
    return ok


# =====================================================================
# 配置
# =====================================================================
CFG = {
    "receptor_dir": "",
    "ligand_parent": "",

    # GPU
    "use_gpu": False,
    "vina_gpu_kernel_dir": "",
    "gpu_thread": 2000,
    "gpu_max_box_size": 25.0,
    "gpu_fallback_cpu": True,

    # ★ 盲对接模式：single | grid | pocket_guided
    "blind_mode": "pocket_guided",

    # 单盒模式
    "n_seeds": 3,
    "exhaustiveness_blind": 64,
    "num_modes_blind": 50,
    "blind_padding": 8.0,
    "blind_max_size": 25.0,

    # 网格模式
    "grid_box_size": 40.0,
    "grid_step": 30.0,
    "grid_probe_ligands": 1,
    "grid_coarse_exh": 4,
    "grid_coarse_seed_count": 1,
    "grid_coarse_num_modes": 5,
    "grid_top_sites": 8,
    "grid_min_gap": 12.0,
    "grid_min_box_atoms": 20,
    "grid_cpus": max(1, multiprocessing.cpu_count()),

    # ★ 口袋引导模式（Rank Fusion）
    "pg_wsl_env": "fpocket_env",       # WSL conda 环境名
    "pg_min_volume": 100.0,             # 口袋最小体积 Å³
    "pg_max_pockets": 20,               # 最多取多少个口袋
    "pg_quick_exh": 4,                  # 快速对接的 exhaustiveness
    "pg_quick_modes": 3,                # 快速对接输出 pose 数
    "pg_rank_lambda": 0.5,              # Rank Fusion λ（0=纯energy，大=偏score）
    "pg_box_size": 25.0,                # 精修盒子边长

    # 精对接
    "exhaustiveness_fine": 32,
    "num_modes_fine": 20,

    # 聚类
    "eps": 5.0,
    "min_samples": 3,
    "min_ratio": 0.4,
    "box_size": 25.0,

    # 开关
    "run_fine": True,
    "export_csv": True,
    "analyze_scope": "all",
    "top_n": 10,

    # 系综
    "ensemble_mode": False,
    "n_conformers": 5,
    "md_traj_dir": "",
    "ensemble_consensus_min": 2,

    # 共识盒子
    "consensus_box": False,             # pocket_guided 模式下不需要
    "box_pose_weight": 0.5,
    "fpocket_bin": "fpocket",
    "p2rank_bin": "prank",
    "box_n_pockets": 3,
    "enable_fpocket": False,
    "enable_p2rank": False,

    # 共识打分
    "consensus_scoring": False,
    "smina_bin": "smina",
    "gnina_bin": "gnina",
    "enable_smina": False,
    "enable_gnina": False,
    "score_weights": {"vina": 0.3, "smina": 0.2, "gnina": 0.5},

    "ligand_ext": ".pdb",
}


# =====================================================================
# PDB → PDBQT
# =====================================================================
def _clean_pdbqt_headers(pdbqt_path):
    try:
        with open(pdbqt_path) as f:
            lines = f.readlines()
    except OSError:
        return False
    bad = ("COMPND", "AUTHOR", "REVDAT", "JRNL", "SEQRES",
           "HEADER", "TITLE", "CRYST1", "ORIGX", "SCALE",
           "MASTER", "END   ", "REMARK", "CONECT")
    out = [l for l in lines if not l.startswith(bad)]
    with open(pdbqt_path, "w") as f:
        f.writelines(out)
    return True


def pdb_to_pdbqt_receptor(pdb_path, pdbqt_path):
    if not OBABEL_BIN:
        raise RuntimeError("未找到 obabel")
    cmd = [OBABEL_BIN, pdb_path, "-O", pdbqt_path, "-xr", "-h", "--delete", "HOH"]
    subprocess.run(cmd, capture_output=True, text=True)
    if os.path.isfile(pdbqt_path) and os.path.getsize(pdbqt_path) > 100:
        _clean_pdbqt_headers(pdbqt_path)
    return os.path.isfile(pdbqt_path) and os.path.getsize(pdbqt_path) > 100


def pdb_to_pdbqt_ligand(pdb_path, pdbqt_path):
    """优先用 Meeko（从SDF），失败则回退 obabel"""
    if MEEKO_BIN:
        sdf_candidate = Path(pdb_path).with_suffix(".sdf")
        if sdf_candidate.is_file():
            r = subprocess.run([MEEKO_BIN, "-i", str(sdf_candidate),
                                "-o", pdbqt_path],
                               capture_output=True, text=True, timeout=120)
            if (os.path.isfile(pdbqt_path)
                    and os.path.getsize(pdbqt_path) > 100):
                return True

    if not OBABEL_BIN:
        return False
    cmd = [OBABEL_BIN, os.path.abspath(pdb_path),
           "-O", os.path.abspath(pdbqt_path), "-h"]
    subprocess.run(cmd, capture_output=True, text=True)
    if not (os.path.isfile(pdbqt_path) and os.path.getsize(pdbqt_path) > 100):
        return False
    _clean_pdbqt_headers(pdbqt_path)
    return os.path.isfile(pdbqt_path) and os.path.getsize(pdbqt_path) > 100


# =====================================================================
# 盒子计算
# =====================================================================
def compute_bounding_box(pdbqt_path, padding, max_size=60.0):
    xs, ys, zs = [], [], []
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
                except ValueError:
                    continue
    if not xs:
        return None, None
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    cz = (max(zs) + min(zs)) / 2
    sx = min(max(max(xs) - min(xs) + padding, 20.0), max_size)
    sy = min(max(max(ys) - min(ys) + padding, 20.0), max_size)
    sz = min(max(max(zs) - min(zs) + padding, 20.0), max_size)
    return (cx, cy, cz), (sx, sy, sz)


def read_box_file(box_path):
    with open(box_path) as f:
        parts = f.read().split()
    return tuple(map(float, parts[:3])), tuple(map(float, parts[3:6]))


# =====================================================================
# Vina 调用
# =====================================================================
def run_vina(receptor_pdbqt, ligand_pdbqt, out_pdbqt, center, size,
             exhaustiveness, num_modes, seed):
    if not VINA_BIN:
        raise RuntimeError("未找到 vina")
    cmd = [
        VINA_BIN,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--out", out_pdbqt,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{size[0]:.3f}",
        "--size_y", f"{size[1]:.3f}",
        "--size_z", f"{size[2]:.3f}",
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--seed", str(seed),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return False
    if r.returncode != 0 or not os.path.isfile(out_pdbqt):
        return False
    return True


# =====================================================================
# Pose 解析
# =====================================================================
def parse_poses(pdbqt_path):
    poses, atoms, energy = [], [], None
    try:
        with open(pdbqt_path) as f:
            for line in f:
                if line.startswith("MODEL"):
                    if atoms:
                        poses.append({"energy": energy, "atoms": atoms})
                    atoms, energy = [], None
                elif line.startswith("ENDMDL"):
                    if atoms:
                        poses.append({"energy": energy, "atoms": atoms})
                    atoms, energy = [], None
                elif line.startswith("REMARK VINA RESULT:"):
                    try:
                        energy = float(line.split()[3])
                    except (IndexError, ValueError):
                        energy = None
                elif line.startswith(("ATOM", "HETATM")):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        elem = line[76:78].strip() if len(line) > 76 else ""
                        if elem != "H":
                            atoms.append((x, y, z))
                    except ValueError:
                        continue
        if atoms:
            poses.append({"energy": energy, "atoms": atoms})
    except OSError:
        pass
    return poses


def read_best_energy(pdbqt_path):
    best = None
    try:
        with open(pdbqt_path) as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    try:
                        e = float(line.split()[3])
                        if best is None or e < best:
                            best = e
                    except (IndexError, ValueError):
                        pass
    except OSError:
        pass
    return best


def centroid(atoms):
    if not atoms:
        return None
    return (sum(a[0] for a in atoms) / len(atoms),
            sum(a[1] for a in atoms) / len(atoms),
            sum(a[2] for a in atoms) / len(atoms))


def pdb_centroid(pdb_path):
    xs, ys, zs = [], [], []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        xs.append(float(line[30:38]))
                        ys.append(float(line[38:46]))
                        zs.append(float(line[46:54]))
                    except ValueError:
                        continue
    except OSError:
        return None
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


def load_all_poses(root, conformer=None):
    all_poses = []
    for fp in sorted(Path(root).rglob("docking_*.pdbqt")):
        for p in parse_poses(str(fp)):
            c = centroid(p["atoms"])
            if c:
                item = {"file": fp.parent.name + "/" + fp.name,
                        "energy": p["energy"], "center": c}
                if conformer is not None:
                    item["conformer"] = conformer
                all_poses.append(item)
    return all_poses


def cluster_poses(all_poses, eps, min_samples):
    if not all_poses:
        return [], None
    X = np.array([p["center"] for p in all_poses])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    clusters = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        idx = [i for i, l in enumerate(labels) if l == lab]
        pts = X[idx]
        energies = [all_poses[i]["energy"] for i in idx
                    if all_poses[i]["energy"] is not None]
        clusters.append({
            "label": int(lab), "size": len(idx),
            "center": pts.mean(axis=0),
            "span": pts.max(axis=0) - pts.min(axis=0),
            "min_energy": min(energies) if energies else None,
            "mean_energy": sum(energies) / len(energies) if energies else None,
        })
    clusters.sort(key=lambda c: (
        -c["size"],
        c["mean_energy"] if c["mean_energy"] is not None else 0))
    return clusters, labels


def annotate_cluster_conformers(clusters, all_poses, labels):
    if labels is None:
        return clusters
    for c in clusters:
        idx = [i for i, l in enumerate(labels) if l == c["label"]]
        confs = [all_poses[i].get("conformer", "crystal") for i in idx]
        dist = {}
        for k in confs:
            dist[k] = dist.get(k, 0) + 1
        c["conformer_dist"] = dist
        c["n_conformers"] = len(dist)
    return clusters


def judge_quality(clusters, n_total, min_ratio):
    if not clusters or n_total == 0:
        return False, "无聚类"
    top_ratio = clusters[0]["size"] / n_total
    msgs = []
    if top_ratio < min_ratio:
        msgs.append(f"top1 {top_ratio:.1%} < {min_ratio:.0%}")
    return (False, "; ".join(msgs)) if msgs else (True, f"top1 {top_ratio:.1%}")


def judge_ensemble_consensus(clusters, min_confs):
    if not clusters:
        return False, "无聚类"
    top = clusters[0]
    n = top.get("n_conformers", 1)
    dist = top.get("conformer_dist", {})
    detail = ", ".join(f"{k}×{v}" for k, v in dist.items())
    return (n >= min_confs), f"top1 覆盖 {n} 构象 ({detail})"


def prepare_ensembles(receptor_pdb, lig_dir):
    if not CFG["ensemble_mode"]:
        return [("crystal", receptor_pdb)]
    md_dir = CFG["md_traj_dir"]
    if not os.path.isdir(md_dir):
        return [("crystal", receptor_pdb)]
    reps = sorted(Path(md_dir).glob("representative_*.pdb"))
    if not reps:
        reps = sorted(Path(md_dir).glob("*.pdb"))
    if not reps:
        return [("crystal", receptor_pdb)]
    n = CFG["n_conformers"]
    pairs = [(f"conf{i}", str(p)) for i, p in enumerate(reps[:n], 1)]
    pairs.append(("crystal", receptor_pdb))
    return pairs


# =====================================================================
# ★ 口袋引导模式（Rank Fusion）
# =====================================================================
def _wsl_path(p):
    """Windows 路径 → WSL 路径"""
    p = str(Path(p).resolve()).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def run_fpocket_wsl(receptor_pdb, out_dir):
    """Linux 原生调用 fpocket（函数名保留兼容，实际不用 WSL）"""
    import platform
    if platform.system() == "Windows":
        # Windows 走 WSL
        p = str(Path(receptor_pdb).resolve()).replace("\\", "/")
        if len(p) >= 2 and p[1] == ":":
            p = f"/mnt/{p[0].lower()}{p[2:]}"
        rec_wsl = p
        out_wsl = _wsl_path(out_dir)
        env = CFG["pg_wsl_env"]
        cmd = ["wsl", "-e", "bash", "-c",
               f"source $HOME/anaconda3/bin/activate {env} && "
               f"mkdir -p '{out_wsl}' && cd '{out_wsl}' && "
               f"fpocket -f '{rec_wsl}'"]
    else:
        # Linux 原生
        os.makedirs(out_dir, exist_ok=True)
        cmd = ["bash", "-c",
               f"cd '{out_dir}' && fpocket -f '{receptor_pdb}'"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=300, encoding="utf-8", errors="replace")
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def parse_fpocket_info(info_path):
    """解析 receptor_info.txt，返回 {pocket_id: {score, volume}}"""
    info = {}
    current = None
    with open(info_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Pocket") and line.endswith(":"):
                current = line.split()[1]
                info[f"pocket{current}"] = {"score": 0.0, "volume": 0.0}
            elif current:
                if line.startswith("Druggability Score"):
                    try:
                        info[f"pocket{current}"]["score"] = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif line.startswith("Volume :"):
                    try:
                        info[f"pocket{current}"]["volume"] = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
    return info


def load_fpocket_pockets(case_dir, min_volume=100.0, max_pockets=20):
    """加载 fpocket 结果，返回口袋列表"""
    out_dir = case_dir / "receptor_out"
    info_path = out_dir / "receptor_info.txt"
    pocket_dir = out_dir / "pockets"
    if not info_path.is_file() or not pocket_dir.is_dir():
        return []

    meta = parse_fpocket_info(info_path)
    pockets = []
    for atm in sorted(pocket_dir.glob("pocket*_atm.pdb"),
                      key=lambda p: int(p.stem.split("_")[0].replace("pocket", ""))):
        key = atm.stem.split("_")[0]
        if key not in meta:
            continue
        vol = meta[key]["volume"]
        if vol < min_volume:
            continue
        xs, ys, zs = [], [], []
        with open(atm) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        xs.append(float(line[30:38]))
                        ys.append(float(line[38:46]))
                        zs.append(float(line[46:54]))
                    except ValueError:
                        continue
        if not xs:
            continue
        c = (float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs)))
        pockets.append({
            "pocket": key,
            "center": c,
            "score": meta[key]["score"],
            "volume": vol,
        })
    # 按 druggability 排序，取前 N 个
    pockets.sort(key=lambda p: p["score"], reverse=True)
    return pockets[:max_pockets]


def _dock_one_pocket(args):
    """单个口袋快速对接，返回 (pocket_key, energy)"""
    (rec_pdbqt, lig_pdbqt, center, out_path, pocket_key) = args
    box_size = CFG["pg_box_size"]
    cmd = [
        VINA_BIN,
        "--receptor", rec_pdbqt,
        "--ligand", lig_pdbqt,
        "--out", out_path,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{box_size}",
        "--size_y", f"{box_size}",
        "--size_z", f"{box_size}",
        "--exhaustiveness", str(CFG["pg_quick_exh"]),
        "--num_modes", str(CFG["pg_quick_modes"]),
        "--seed", "42",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return (pocket_key, None)
    return (pocket_key, read_best_energy(out_path))


def run_pocket_guided(receptor_pdb, ligand_dir, out_dir):
    """
    ★ 口袋引导对接：
    1. fpocket 找口袋
    2. 每个口袋快速对接
    3. Rank Fusion 选最佳口袋
    4. 在最佳盒子里精修
    """
    work = os.path.join(out_dir, "work")
    os.makedirs(work, exist_ok=True)

    # 1. 受体准备
    rec_pdbqt = os.path.join(work, "receptor.pdbqt")
    if not os.path.isfile(rec_pdbqt) or os.path.getsize(rec_pdbqt) < 100:
        print(f"    转换受体 → pdbqt")
        if not pdb_to_pdbqt_receptor(receptor_pdb, rec_pdbqt):
            print("    ❌ 受体转换失败")
            return False, []

    # 2. fpocket
    print(f"    ▶ fpocket 找口袋...")
    fpocket_out = os.path.join(work, "fpocket")
    os.makedirs(fpocket_out, exist_ok=True)
    # fpocket 输出到输入文件所在目录，所以先复制受体 pdb
    rec_local = os.path.join(fpocket_out, "receptor.pdb")
    if not os.path.isfile(rec_local):
        shutil.copy(receptor_pdb, rec_local)
    ok = run_fpocket_wsl(rec_local, fpocket_out)
    if not ok:
        print("    ⚠ fpocket 失败，回退 grid 模式")
        return False, []

    case_dir = Path(fpocket_out)
    pockets = load_fpocket_pockets(case_dir, CFG["pg_min_volume"], CFG["pg_max_pockets"])
    if not pockets:
        print("    ⚠ fpocket 无有效口袋")
        return False, []
    print(f"    ✓ fpocket 找到 {len(pockets)} 个口袋")

    # 3. 配体准备
    ligs = sorted(Path(ligand_dir).glob(f"*{CFG['ligand_ext']}"))
    if not ligs:
        print("    ⚠ 无配体")
        return False, []

    lig_work = os.path.join(work, "ligands")
    os.makedirs(lig_work, exist_ok=True)
    lig_pdbqts = []
    for lig in ligs:
        out_pdbqt = os.path.join(lig_work, f"{lig.stem}.pdbqt")
        if not os.path.isfile(out_pdbqt) or os.path.getsize(out_pdbqt) < 100:
            if not pdb_to_pdbqt_ligand(str(lig), out_pdbqt):
                continue
        lig_pdbqts.append(out_pdbqt)

    if not lig_pdbqts:
        print("    ⚠ 无有效配体")
        return False, []
    print(f"    配体: {len(lig_pdbqts)}")

    # 用最小的配体作为 probe
    def count_atoms(p):
        try:
            return sum(1 for l in open(p) if l.startswith(("ATOM", "HETATM")))
        except OSError:
            return 0
    lig_pdbqts.sort(key=count_atoms)
    probe = lig_pdbqts[0]
    print(f"    probe: {Path(probe).stem}")

    # 4. 每个口袋快速对接
    print(f"    ★ 快速对接 {len(pockets)} 个口袋 (exh={CFG['pg_quick_exh']})...")
    tasks = []
    for p in pockets:
        out_pdbqt = os.path.join(work, f"quick_{p['pocket']}.pdbqt")
        tasks.append((rec_pdbqt, probe, p["center"], out_pdbqt, p["pocket"]))

    results = {}
    n_cpus = CFG["grid_cpus"]
    with ProcessPoolExecutor(max_workers=n_cpus) as ex:
        futures = {ex.submit(_dock_one_pocket, t): t[4] for t in tasks}
        for fut in as_completed(futures):
            key, e = fut.result()
            results[key] = e

    # 5. Rank Fusion
    valid = [p for p in pockets if results.get(p["pocket"]) is not None]
    if not valid:
        print("    ⚠ 所有口袋对接失败")
        return False, []

    by_e = sorted(valid, key=lambda p: results[p["pocket"]])
    by_s = sorted(valid, key=lambda p: -p["score"])
    rank_e = {p["pocket"]: i for i, p in enumerate(by_e)}
    rank_s = {p["pocket"]: i for i, p in enumerate(by_s)}
    lam = CFG["pg_rank_lambda"]
    best = min(valid, key=lambda p: rank_e[p["pocket"]] + lam * rank_s[p["pocket"]])

    print(f"    ✓ Rank Fusion (λ={lam}) 选中 {best['pocket']}  "
          f"(score={best['score']:.3f}, E={results[best['pocket']]:.2f})")

    # >>> AF3 DUMP: 写 pocket_ranks.json 供 af3_confidence_fusion.py 使用
    _dump_pocket_ranks(valid, results, out_dir)
    # <<< AF3 DUMP

    final_center = best["center"]
    box_size = CFG["pg_box_size"]

    # 6. 精修：在最佳盒子里跑所有配体
    fine_dir = os.path.join(out_dir, "docking_results")
    os.makedirs(fine_dir, exist_ok=True)
    print(f"    ★ 精修 (exh={CFG['exhaustiveness_fine']})...")
    ok_n = 0
    for lig_i, lig_pdbqt in enumerate(lig_pdbqts, 1):
        lig_name = Path(lig_pdbqt).stem
        out_pdbqt = os.path.join(fine_dir, f"docking_{lig_name}.pdbqt")
        if os.path.isfile(out_pdbqt) and os.path.getsize(out_pdbqt) > 100:
            ok_n += 1
            continue
        print(f"      ({lig_i}/{len(lig_pdbqts)}) {lig_name} ...", end="", flush=True)
        if run_vina(rec_pdbqt, lig_pdbqt, out_pdbqt, final_center,
                    (box_size, box_size, box_size),
                    CFG["exhaustiveness_fine"], CFG["num_modes_fine"], 42):
            ok_n += 1
            print(" ✓")
        else:
            print(" ✗")

    # 写 box 文件
    box_path = os.path.join(os.path.dirname(receptor_pdb),
                            os.path.splitext(os.path.basename(receptor_pdb))[0] + ".box")
    with open(box_path, "w") as f:
        f.write(f"{final_center[0]:.3f} {final_center[1]:.3f} {final_center[2]:.3f} "
                f"{box_size:.3f} {box_size:.3f} {box_size:.3f}\n")
    print(f"    ✅ {box_path}")

    return True, [("rank_fusion", final_center)]


# =====================================================================
# >>> AF3 FUSION DUMP
# =====================================================================
def _dump_pocket_ranks(valid_pockets, energy_map, out_dir):
    """
    把 Rank Fusion 用到的候选口袋 dump 成 pocket_ranks.json。
    valid_pockets : list[dict]，每项有 'pocket' / 'center' / 'score'
    energy_map    : dict[pocket_key] -> quick vina energy
    out_dir       : 写到这个目录下的 pocket_ranks.json
    """
    dump = []
    for i, p in enumerate(valid_pockets):
        c = p.get("center")
        if c is None:
            continue
        try:
            center_list = [float(c[0]), float(c[1]), float(c[2])]
        except Exception:
            continue

        try:
            drug = float(p.get("score", 0.0))
        except Exception:
            drug = 0.0

        e = energy_map.get(p["pocket"]) if energy_map else None
        try:
            e = float(e) if e is not None else 0.0
        except Exception:
            e = 0.0

        dump.append({
            "index":        i,
            "pocket":       p["pocket"],
            "center":       center_list,
            "druggability": drug,
            "vina_energy":  e,
        })

    if not dump:
        print("    ⚠ pocket_ranks.json: 没有可 dump 的口袋")
        return

    try:
        out = Path(out_dir) / "pocket_ranks.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)
        print(f"    ✓ pocket_ranks.json ({len(dump)} pockets) → {out}")
    except Exception as e:
        print(f"    ⚠ pocket_ranks.json 写入失败: {e}")
# =====================================================================
# <<< AF3 FUSION DUMP
# =====================================================================


# =====================================================================
# 单盒模式 / 网格模式（保留原逻辑）
# =====================================================================
def run_blind_single(receptor_pdb, ligand_dir, seed, out_dir=None):
    """简化版：bbox + max_size"""
    if out_dir is None:
        out_dir = ligand_dir
    work = os.path.join(out_dir, "work")
    os.makedirs(work, exist_ok=True)
    out_docking = os.path.join(out_dir, "docking_results")
    os.makedirs(out_docking, exist_ok=True)

    rec_pdbqt = os.path.join(work, "receptor.pdbqt")
    if not os.path.isfile(rec_pdbqt) or os.path.getsize(rec_pdbqt) < 100:
        if not pdb_to_pdbqt_receptor(receptor_pdb, rec_pdbqt):
            return False

    center, size = compute_bounding_box(rec_pdbqt, CFG["blind_padding"],
                                         CFG["blind_max_size"])
    if not center:
        return False

    ligs = sorted(Path(ligand_dir).glob(f"*{CFG['ligand_ext']}"))
    lig_pdbqts = []
    for lig in ligs:
        lig_pdbqt = os.path.join(work, f"{lig.stem}.pdbqt")
        if not os.path.isfile(lig_pdbqt) or os.path.getsize(lig_pdbqt) < 100:
            if not pdb_to_pdbqt_ligand(str(lig), lig_pdbqt):
                continue
        lig_pdbqts.append(lig_pdbqt)

    ok_n = 0
    for lig_pdbqt in lig_pdbqts:
        lig_name = Path(lig_pdbqt).stem
        out_pdbqt = os.path.join(out_docking, f"docking_{lig_name}.pdbqt")
        if os.path.isfile(out_pdbqt) and os.path.getsize(out_pdbqt) > 100:
            ok_n += 1
            continue
        if run_vina(rec_pdbqt, lig_pdbqt, out_pdbqt, center, size,
                    CFG["exhaustiveness_blind"], CFG["num_modes_blind"], seed):
            ok_n += 1
    return ok_n > 0


# =====================================================================
# 精对接
# =====================================================================
def run_fine_native(receptor_pdb, ligand_dir, box_path):
    work = os.path.join(ligand_dir, "work_fine")
    os.makedirs(work, exist_ok=True)
    out_docking = os.path.join(ligand_dir, "docking_results")
    if os.path.isdir(out_docking):
        shutil.rmtree(out_docking)
    os.makedirs(out_docking, exist_ok=True)

    rec_pdbqt = os.path.join(work, "receptor.pdbqt")
    if not os.path.isfile(rec_pdbqt) or os.path.getsize(rec_pdbqt) < 100:
        if not pdb_to_pdbqt_receptor(receptor_pdb, rec_pdbqt):
            return False

    center, size = read_box_file(box_path)
    ligs = sorted(Path(ligand_dir).glob(f"*{CFG['ligand_ext']}"))
    ok_n = 0
    for lig in ligs:
        lig_pdbqt = os.path.join(work, f"{lig.stem}.pdbqt")
        if not os.path.isfile(lig_pdbqt) or os.path.getsize(lig_pdbqt) < 100:
            if not pdb_to_pdbqt_ligand(str(lig), lig_pdbqt):
                continue
        out_pdbqt = os.path.join(out_docking, f"docking_{lig.stem}.pdbqt")
        if run_vina(rec_pdbqt, lig_pdbqt, out_pdbqt, center, size,
                    CFG["exhaustiveness_fine"], CFG["num_modes_fine"], 42):
            ok_n += 1
    return ok_n > 0


# =====================================================================
# CSV 导出
# =====================================================================
def export_hotspot_csv(lig_dir, all_poses, clusters, labels):
    if not CFG["export_csv"]:
        return
    csv1 = os.path.join(lig_dir, "hotspot_clusters.csv")
    with open(csv1, "w", newline="", encoding="utf-8") as f:
        f.write("Cluster,Poses,Pct,Center_X,Center_Y,Center_Z,"
                "Span_X,Span_Y,Span_Z,Min_Energy,Mean_Energy\n")
        for i, c in enumerate(clusters, 1):
            pct = 100 * c["size"] / len(all_poses)
            f.write(f"{i},{c['size']},{pct:.2f},{c['center'][0]:.3f},"
                    f"{c['center'][1]:.3f},{c['center'][2]:.3f},"
                    f"{c['span'][0]:.3f},{c['span'][1]:.3f},{c['span'][2]:.3f},"
                    f"{c['min_energy'] or ''},{c['mean_energy'] or ''}\n")
    print(f"  ✅ {csv1}")


def export_fine_csv(lig_dir):
    if not CFG["export_csv"]:
        return
    fine_dir = os.path.join(lig_dir, "docking_results")
    if not os.path.isdir(fine_dir):
        return
    rows = []
    for fp in sorted(Path(fine_dir).glob("docking_*.pdbqt")):
        poses = parse_poses(str(fp))
        if not poses:
            continue
        energies = [p["energy"] for p in poses if p["energy"] is not None]
        best = min(energies) if energies else None
        rows.append((fp.stem.replace("docking_", ""), best, len(poses)))
    rows.sort(key=lambda r: r[1] if r[1] is not None else 999)
    path = os.path.join(lig_dir, "fine_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("Ligand,Best_Energy_kcal_mol,Num_Poses\n")
        for n, b, c in rows:
            f.write(f"{n},{f'{b:.3f}' if b is not None else ''},{c}\n")
    print(f"  ✅ {path}")


def archive_results(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    if os.path.exists(src):
        shutil.move(src, dst)


# =====================================================================
# 主流程
# =====================================================================
def process_receptor(receptor_pdb, ligand_parent):
    name = os.path.splitext(os.path.basename(receptor_pdb))[0]
    lig_dir = os.path.join(ligand_parent, name)

    print(f"\n{'=' * 70}\n[{name}]")
    print(f"  受体: {receptor_pdb}")
    print(f"  配体: {lig_dir}")
    print(f"  模式: {CFG['blind_mode']}")
    print(f"{'=' * 70}")

    if not os.path.isdir(lig_dir):
        print(f"  ❌ 配体目录不存在")
        return False

    if CFG["blind_mode"] == "pocket_guided":
        # ★ 口袋引导模式
        ok, _ = run_pocket_guided(receptor_pdb, lig_dir, lig_dir)
        if not ok:
            print("  ❌ 口袋引导失败")
            return False
        export_fine_csv(lig_dir)
        return True

    elif CFG["blind_mode"] == "single":
        tmp_dir = os.path.join(lig_dir, "_tmp_single")
        for seed in range(1, CFG["n_seeds"] + 1):
            print(f"    seed={seed}")
            run_blind_single(receptor_pdb, lig_dir, seed, out_dir=tmp_dir)
            dst = os.path.join(lig_dir, f"blind_seed_{seed}")
            archive_results(os.path.join(tmp_dir, "docking_results"), dst)
            shutil.rmtree(tmp_dir, ignore_errors=True)

        all_poses = []
        for d in Path(lig_dir).glob("blind_seed_*"):
            all_poses.extend(load_all_poses(str(d), conformer="crystal"))
        if not all_poses:
            return False
        clusters, labels = cluster_poses(all_poses, CFG["eps"], CFG["min_samples"])
        clusters = annotate_cluster_conformers(clusters, all_poses, labels)
        if not clusters:
            return False
        top = clusters[0]
        center = tuple(top["center"])
        bs = CFG["box_size"]
        box_path = os.path.join(os.path.dirname(receptor_pdb), name + ".box")
        with open(box_path, "w") as f:
            f.write(f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f} "
                    f"{bs:.3f} {bs:.3f} {bs:.3f}\n")
        export_hotspot_csv(lig_dir, all_poses, clusters, labels)
        if CFG["run_fine"]:
            run_fine_native(receptor_pdb, lig_dir, box_path)
            export_fine_csv(lig_dir)
        return True

    else:
        print(f"  ⚠ grid 模式需要完整代码（见之前的版本）")
        return False


# =====================================================================
# 批量入口
# =====================================================================
def _list_pdbs(d):
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith(".pdb")
                  and os.path.isfile(os.path.join(d, f)))


def run_all():
    rec_dir, lig_parent = CFG["receptor_dir"], CFG["ligand_parent"]
    if not os.path.isdir(rec_dir) or not os.path.isdir(lig_parent):
        print("❌ 目录无效")
        return
    pdbs = _list_pdbs(rec_dir)
    if not pdbs:
        print("❌ 无 .pdb")
        return
    print(f"\n发现 {len(pdbs)} 受体")
    ok_n = fail_n = 0
    for p in pdbs:
        try:
            if process_receptor(os.path.join(rec_dir, p), lig_parent):
                ok_n += 1
            else:
                fail_n += 1
        except Exception:
            import traceback
            traceback.print_exc()
            fail_n += 1
    print(f"\n完成: {ok_n} 成功 / {fail_n} 失败")


# =====================================================================
# 菜单
# =====================================================================
def choose_folder(title):
    try:
        import tkinter as tk
        from tkinter import filedialog
        r = tk.Tk()
        r.withdraw()
        d = filedialog.askdirectory(title=title)
        r.destroy()
        if d:
            return d
    except Exception:
        pass
    return input(f"{title}（路径）: ").strip()


def menu():
    print("\n" + "=" * 70)
    print("  全自动对接（单盒 / 网格 / ★口袋引导 Rank Fusion）")
    print("=" * 70)
    print(f"  大分子:   {CFG['receptor_dir'] or '(未设置)'}")
    print(f"  小分子:   {CFG['ligand_parent'] or '(未设置)'}")
    print(f"  盲模式:   {CFG['blind_mode']}")
    if CFG["blind_mode"] == "pocket_guided":
        print(f"    fpocket: min_vol={CFG['pg_min_volume']}Å³ "
              f"max_pockets={CFG['pg_max_pockets']}")
        print(f"    快接:   exh={CFG['pg_quick_exh']} modes={CFG['pg_quick_modes']}")
        print(f"    Rank Fusion: λ={CFG['pg_rank_lambda']} "
              f"box={CFG['pg_box_size']}Å")
    print(f"  精对接:   exh={CFG['exhaustiveness_fine']} "
          f"modes={CFG['num_modes_fine']}")
    print(f"  工具:     vina={'✓' if VINA_BIN else '✗'}  "
          f"obabel={'✓' if OBABEL_BIN else '✗'}")
    print("=" * 70)
    print("  1. 选大分子文件夹")
    print("  2. 选小分子父文件夹")
    print("  3. 调整参数")
    print("  4. ▶ 全自动运行")
    print("  0. 退出")
    print("=" * 70)


def adjust_params():
    print()
    s = input(f"  模式 single/grid/pocket_guided（当前 {CFG['blind_mode']}）: ").strip().lower()
    if s in ("single", "grid", "pocket_guided"):
        CFG["blind_mode"] = s

    if CFG["blind_mode"] == "pocket_guided":
        print("\n  --- 口袋引导模式 ---")
        fields = [
            ("pg_min_volume", "口袋最小体积 Å³", float),
            ("pg_max_pockets", "最多口袋数", int),
            ("pg_quick_exh", "快速对接 exh", int),
            ("pg_quick_modes", "快速对接 modes", int),
            ("pg_rank_lambda", "Rank Fusion λ", float),
            ("pg_box_size", "精修盒子边长", float),
            ("pg_wsl_env", "WSL conda 环境名", str),
        ]
        for k, lab, t in fields:
            v = input(f"  {lab}（当前 {CFG[k]}）: ").strip()
            if v:
                try:
                    CFG[k] = t(v)
                except ValueError:
                    pass

    print("\n  --- 精对接 ---")
    for k, lab, t in [("exhaustiveness_fine", "exh", int),
                      ("num_modes_fine", "modes", int)]:
        v = input(f"  {lab}（当前 {CFG[k]}）: ").strip()
        if v:
            try:
                CFG[k] = t(v)
            except ValueError:
                pass


def main():
    if not VINA_BIN:
        print("❌ 未找到 vina")
        sys.exit(1)
    while True:
        menu()
        c = input("请选择: ").strip()
        if c == "0":
            return
        elif c == "1":
            d = choose_folder("选择大分子文件夹")
            if d:
                CFG["receptor_dir"] = d
                print(f"✅ {d}")
        elif c == "2":
            d = choose_folder("选择小分子父文件夹")
            if d:
                CFG["ligand_parent"] = d
                print(f"✅ {d}")
        elif c == "3":
            adjust_params()
        elif c == "4":
            if not CFG["receptor_dir"] or not CFG["ligand_parent"]:
                print("❌ 先设置路径")
                continue
            run_all()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()