#!/usr/bin/env python3
"""一键检查所有依赖是否齐全。放到项目目录直接跑：python check_env.py"""
import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OK, MISSING = [], []


def line(msg=""):
    print(msg)


def check_py_module(name, import_name=None, extra_msg=""):
    import_name = import_name or name
    try:
        __import__(import_name)
        OK.append(f"[Py] {name}")
        line(f"  ✓ {name} {extra_msg}")
    except ImportError:
        MISSING.append(f"[Py] {name}  →  pip install {name}")
        line(f"  ✗ {name}   ← 缺，pip install {name}")


def check_exe(name, manual_path=None):
    p = shutil.which(name)
    if not p and manual_path and os.path.isfile(manual_path):
        p = manual_path
    if p:
        OK.append(f"[Exe] {name} -> {p}")
        line(f"  ✓ {name}  ->  {p}")
        return p
    else:
        MISSING.append(f"[Exe] {name}")
        line(f"  ✗ {name}   ← 不在 PATH")
        return None


def check_wsl_fpocket(env_name="fpocket_env"):
    if platform.system() != "Windows":
        p = shutil.which("fpocket")
        if p:
            OK.append("[Exe] fpocket")
            line(f"  ✓ fpocket (Linux) -> {p}")
        else:
            MISSING.append("[Exe] fpocket")
            line("  ✗ fpocket 未安装")
        return
    # Windows：查 WSL
    line("  · Windows 环境，检查 WSL + fpocket ...")
    try:
        r = subprocess.run(
            ["wsl", "-e", "bash", "-lc",
             f"source $HOME/miniconda3/bin/activate {env_name} 2>/dev/null && "
             f"which fpocket"],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace")
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        if "fpocket" in out:
            OK.append(f"[WSL] fpocket in {env_name}")
            line(f"  ✓ WSL fpocket -> {out}")
        else:
            MISSING.append(f"[WSL] fpocket (env={env_name})")
            line(f"  ✗ WSL 里没找到 fpocket (env={env_name})")
            if err:
                line(f"     stderr: {err[:200]}")
    except FileNotFoundError:
        MISSING.append("[WSL] wsl 命令")
        line("  ✗ 没有 wsl 命令")
    except subprocess.TimeoutExpired:
        MISSING.append("[WSL] fpocket (timeout)")
        line("  ✗ WSL 调用超时")


def check_py_file(path, label):
    p = Path(path) if os.path.isabs(path) else SCRIPT_DIR / path
    if p.is_file():
        OK.append(f"[File] {label}")
        line(f"  ✓ {label}  ->  {p}")
        return True
    MISSING.append(f"[File] {label}")
    line(f"  ✗ {label}   ← 缺文件：{p}")
    return False


def check_usalign_path():
    """从 af3_common.py 里读 USALIGN_EXE 并验证"""
    p = SCRIPT_DIR / "af3_common.py"
    if not p.is_file():
        line("  · 没找到 af3_common.py，跳过 USalign 检查")
        return
    exe = None
    with open(p, encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if s.startswith("USALIGN_EXE"):
                try:
                    exe = s.split("=", 1)[1].strip().strip('"').strip("'")
                except Exception:
                    pass
                break
    if not exe:
        MISSING.append("[USalign] 无法解析 USALIGN_EXE")
        line("  ✗ af3_common.py 里没找到 USALIGN_EXE 或写法不对")
        return
    if os.path.isfile(exe):
        OK.append(f"[USalign] {exe}")
        line(f"  ✓ USalign -> {exe}")
    else:
        MISSING.append(f"[USalign] 路径不存在: {exe}")
        line(f"  ✗ USalign 路径不存在：{exe}")
        line(f"     → 修改 af3_common.py 顶部的 USALIGN_EXE")


# =====================================================================
line("=" * 70)
line("依赖体检")
line("=" * 70)

line("\n【1】Python 包")
check_py_module("gemmi")
check_py_module("numpy")
check_py_module("scipy")
check_py_module("statsmodels")
check_py_module("sklearn", "sklearn")
check_py_module("rdkit", "rdkit")

line("\n【2】外部可执行文件")
check_exe("vina", r"E:\tools\vina\vina.exe")
check_exe("obabel")
check_exe("mk_prepare_ligand")     # Meeko（可选）

line("\n【3】fpocket")
check_wsl_fpocket("fpocket_env")

line("\n【4】项目内文件")

# scripts/ 的上一级是 ALL-DATA 根目录
ROOT = Path(__file__).resolve().parent.parent

check_py_file(str(ROOT / "src" / "af3_common.py"),
              "src/af3_common.py")
check_py_file(str(ROOT / "pipeline" / "auto_pipeline.py"),
              "pipeline/auto_pipeline.py")
check_py_file(str(ROOT / "pipeline" / "run_af3_rankfusion.py"),
              "pipeline/run_af3_rankfusion.py")
check_py_file(str(ROOT / "pipeline" / "run_experiment_baselines.py"),
              "pipeline/run_experiment_baselines.py")
check_py_file(str(ROOT / "analysis" / "evaluate_boxes.py"),
              "analysis/evaluate_boxes.py")
check_py_file(str(ROOT / "analysis" / "stats_analysis.py"),
              "analysis/stats_analysis.py")

line("\n【5】USalign 路径")
check_usalign_path()

line("\n【6】数据目录")
ROOT = Path(__file__).resolve().parent.parent
for d in ("AF3_workflow", "exp_structures", "ligands"):
    p = ROOT / d
    if p.is_dir():
        n = len([x for x in p.iterdir() if x.is_dir()])
        OK.append(f"[Data] {d}")
        line(f"  ✓ {d}/   ({n} 个 case)")
    else:
        MISSING.append(f"[Data] {d}/")
        line(f"  ✗ {d}/   不存在")

line("\n【7】输出目录")
p = SCRIPT_DIR / "results"
if p.is_dir():
    line(f"  ✓ results/ 已存在")
else:
    line(f"  · results/ 未创建（首次运行会自动建）")

line("\n" + "=" * 70)
line("体检结果")
line("=" * 70)
line(f"\n✓ 通过：{len(OK)} 项")
line(f"✗ 缺失：{len(MISSING)} 项")

if MISSING:
    line("\n需要处理：")
    for m in MISSING:
        line(f"  · {m}")
    line("\n按上面提示补齐即可。")
else:
    line("\n全部通过，可以跑：")
    line("  python af3_postprocess_v2.py --cases <你的case名>")