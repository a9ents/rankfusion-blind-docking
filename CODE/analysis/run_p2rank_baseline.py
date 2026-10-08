#!/usr/bin/env python3
"""run_p2rank_baseline.py — P2Rank 循环单文件跑实验结构，导出 top-1 pocket 中心为 .box。"""
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
import af3_common as ac  # noqa: E402


# ============================================================
# 配置
# ============================================================
P2RANK_DIR = Path(r"E:\AF3环境\p2rank_2.6-alpha")
JAVA_BIN = "java"
WORK_ROOT = Path(r"E:\AF3环境\p2rank_work")

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
    """单个 case 跑 P2Rank。在子进程里执行。

    返回 (case, info_or_None)，info = {"center": np.array, "score": float}
    """
    case, pdb_path, out_root = args
    out_dir = Path(out_root) / case
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cmd = [
        JAVA_BIN,
        "-Dfile.encoding=UTF-8",
        "-cp", build_classpath(),
        *JAVA_OPTS,
        "cz.siret.prank.program.Main",
        "predict",
        "-o", str(out_dir),
        "-f", str(pdb_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    except subprocess.TimeoutExpired:
        return case, None

    # 递归找输出 csv
    csvs = list(out_dir.rglob(f"{Path(pdb_path).stem}.pdb_predictions.csv"))
    if not csvs:
        return case, None

    with open(csvs[0], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v.strip() if isinstance(v, str) else v)
                   for k, v in row.items()}
            try:
                if int(row.get("rank", "999")) != 1:
                    continue
                return case, {
                    "center": np.array([float(row["center_x"]),
                                       float(row["center_y"]),
                                       float(row["center_z"])]),
                    "score": float(row.get("score", "0")),
                }
            except (KeyError, ValueError):
                continue
    return case, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", default="exp_structures")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--cases", default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        cases = sorted([d.name for d in exp_dir.iterdir() if d.is_dir()])
    print(f"共 {len(cases)} 个 case")

    # 准备输入目录（P2Rank 输出目录也要能写）
    tmp_in = WORK_ROOT / "input"
    if tmp_in.exists():
        shutil.rmtree(tmp_in)
    tmp_in.mkdir(parents=True)

    valid_cases = []
    for case in cases:
        rec = exp_dir / case / "receptor.pdb"
        if not rec.is_file():
            print(f"  ⚠ 跳过 {case}: 无 receptor.pdb")
            continue
        shutil.copy(rec, tmp_in / f"{case}.pdb")
        valid_cases.append(case)
    print(f"有效受体: {len(valid_cases)}")

    # 输出目录
    p2rank_out_root = WORK_ROOT / "p2rank_out"
    if p2rank_out_root.exists():
        shutil.rmtree(p2rank_out_root)
    p2rank_out_root.mkdir(parents=True)

    # 准备任务
    tasks = [
        (case, str(tmp_in / f"{case}.pdb"), p2rank_out_root)
        for case in valid_cases
    ]

    print(f"启动 {args.workers} 个并发进程...")
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            case, info = fut.result()
            results[case] = info
            done += 1
            if done % 20 == 0 or done == len(tasks):
                n_ok = sum(1 for v in results.values() if v is not None)
                print(f"  进度 {done}/{len(tasks)}  成功 {n_ok}")

    # 写 box + summary
    rows = []
    for case in valid_cases:
        info = results.get(case)
        if info:
            box_out = out_dir / f"p2rank_top1_{case}.box"
            with open(box_out, "w") as f:
                f.write(f"center_x = {info['center'][0]:.3f}\n")
                f.write(f"center_y = {info['center'][1]:.3f}\n")
                f.write(f"center_z = {info['center'][2]:.3f}\n")
            rows.append({"case": case,
                         "p2rank_score": round(info["score"], 4),
                         "box_written": 1})
        else:
            rows.append({"case": case, "p2rank_score": "", "box_written": 0})

    n_ok = sum(1 for r in rows if r["box_written"] == 1)
    print(f"\n成功: {n_ok}/{len(rows)}")

    csv_path = out_dir / "p2rank_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "p2rank_score", "box_written"])
        w.writeheader()
        w.writerows(rows)
    print(f"✅ {csv_path}")


if __name__ == "__main__":
    main()