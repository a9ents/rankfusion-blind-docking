#!/usr/bin/env python3
"""
run_p2rank_baseline.py — P2Rank 循环单文件跑实验结构
"""
import os
import argparse
import csv
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT / "src"))
import af3_common as ac  # type: ignore # noqa: E402

# ============================================================
# 配置（自适应）
# ============================================================
P2RANK_DIR = Path(os.environ.get(
    "P2RANK_DIR", str(_ROOT / "tools" / "p2rank_2.6-alpha")))
JAVA_BIN = os.environ.get("JAVA_BIN", "java")
WORK_ROOT = Path(os.environ.get(
    "P2RANK_WORK_ROOT", str(_ROOT / "work" / "p2rank_work")))

JAVA_OPTS = [
    "-Xmx2048m",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/jdk.internal.misc=ALL-UNNAMED",
]


def build_classpath():
    jar = P2RANK_DIR / "bin" / "p2rank.jar"
    lib = P2RANK_DIR / "bin" / "lib" / "*"
    return f"{jar};{lib}"


def run_one(args):
    case, pdb_path, out_root = args
    out_dir = Path(out_root) / case
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cmd = [
        JAVA_BIN, "-Dfile.encoding=UTF-8", "-cp", build_classpath(),
        *JAVA_OPTS, "cz.siret.prank.program.Main", "predict", "-o", str(out_dir), "-f", str(pdb_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    except subprocess.TimeoutExpired:
        return case, None

    csvs = list(out_dir.rglob(f"{Path(pdb_path).stem}.pdb_predictions.csv"))
    if not csvs:
        return case, None

    with open(csvs[0], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            try:
                if int(row.get("rank", "999")) != 1:
                    continue
                return case, {
                    "center": np.array([float(row["center_x"]), float(row["center_y"]), float(row["center_z"])]),
                    "score": float(row.get("score", "0")),
                }
            except (KeyError, ValueError):
                continue
    return case, None