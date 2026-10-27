#!/usr/bin/env python3
"""
run_all.py — 一键跑完全流程：
  1. AF3 结构 → Rank Fusion 盒子
  2. 实验结构 3 个 baseline 盒子
  3. 统一评估
  4. 统计检验
  5. （可选）AF3 置信度融合
"""
import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]   # .../ALL-DATA


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(_ROOT))
    if r.returncode != 0:
        print(f"⚠ 命令失败: {' '.join(cmd)}")
    return r.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=None,
                   help="逗号分隔的 case 名；默认全部")
    p.add_argument("--af3_output", default="AF3_workflow")
    p.add_argument("--exp_dir", default="exp_structures")
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--with_fusion", action="store_true",
                   help="额外跑 AF3 置信度融合")
    args = p.parse_args()

    cases_arg = ["--cases", args.cases] if args.cases else []

    # 1. AF3 → Rank Fusion
    run([sys.executable, str(_ROOT / "pipeline" / "run_af3_rankfusion.py"),
         "--af3_output", args.af3_output,
         "--exp_dir", args.exp_dir,
         "--ligand_dir", args.ligand_dir,
         "--out_dir", args.out_dir] + cases_arg)

    # 2. 实验结构基线
    run([sys.executable, str(_ROOT / "pipeline" / "run_experiment_baselines.py"),
         "--exp_dir", args.exp_dir,
         "--ligand_dir", args.ligand_dir,
         "--out_dir", args.out_dir] + cases_arg)

    # 3. 评估
    run([sys.executable, str(_ROOT / "analysis" / "evaluate_boxes.py"),
         "--ligand_dir", args.ligand_dir,
         "--box_dir", args.out_dir,
         "--out_csv", str(Path(args.out_dir) / "evaluation.csv")])

    # 4. 统计
    run([sys.executable, str(_ROOT / "analysis" / "stats_analysis.py"),
         "--eval_csv", str(Path(args.out_dir) / "evaluation.csv"),
         "--out_dir", args.out_dir])

    # 5. 可选：AF3 置信度融合
    if args.with_fusion:
        if args.cases:
            cases = [c.strip() for c in args.cases.split(",")]
        else:
            root = Path(args.af3_output)
            cases = sorted([d.name for d in root.iterdir() if d.is_dir()])
        for case in cases:
            pocket_json = Path(args.out_dir) / case / "af3_top1" / "pocket_ranks.json"
            af3_lig = Path(args.out_dir) / f"af3ligand_{case}.pdb"
            if pocket_json.is_file() and af3_lig.is_file():
                run([sys.executable, str(_ROOT / "src" / "rankfusion_af3.py"),
                     "--pocket_json", str(pocket_json),
                     "--af3_ligand", str(af3_lig),
                     "--out_box", str(Path(args.out_dir) / f"af3fusion_{case}.box")])
            else:
                print(f"  ⚠ 跳过 fusion {case}：缺 pocket_ranks.json 或 af3ligand")

    print("\n✅ 全部流程完成")


if __name__ == "__main__":
    main()