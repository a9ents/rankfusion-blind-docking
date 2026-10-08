#!/usr/bin/env python3
"""
rankfusion_af3.py — 把 AF3 预测配体作为第三排序信号融合。

融合公式：
    combined_rank(i) = rank_E(i) + λ · rank_S(i) + μ · rank_C(i)
其中 rank_C 为口袋质心到 AF3 预测配体质心的距离升序。

用法：
    python rankfusion_af3.py --pocket_json <path> \
        --af3_ligand <path> \
        --out_box <path> \
        [--lambda_ 0.75] [--mu 1.0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 确保无论从哪里运行，都能找到同目录的 af3_common
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import af3_common as ac


def fused_box(pocket_json, af3_ligand_pdb, lam=0.75, mu=1.0):
    with open(pocket_json) as f:
        pockets = json.load(f)

    af3_c = ac.ligand_centroid(af3_ligand_pdb)
    if af3_c is None:
        raise RuntimeError("无法计算 AF3 配体质心")

    # rank_E：能量升序
    order_E = sorted(range(len(pockets)), key=lambda i: pockets[i]["vina_energy"])
    rank_E = [0] * len(pockets)
    for r, i in enumerate(order_E):
        rank_E[i] = r

    # rank_S：druggability 降序
    order_S = sorted(range(len(pockets)), key=lambda i: -pockets[i]["druggability"])
    rank_S = [0] * len(pockets)
    for r, i in enumerate(order_S):
        rank_S[i] = r

    # rank_C：距 AF3 配体距离升序
    dists = [ac.euclidean(p["center"], af3_c) for p in pockets]
    order_C = sorted(range(len(pockets)), key=lambda i: dists[i])
    rank_C = [0] * len(pockets)
    for r, i in enumerate(order_C):
        rank_C[i] = r

    combined = [rank_E[i] + lam * rank_S[i] + mu * rank_C[i]
                for i in range(len(pockets))]
    best = int(np.argmin(combined))
    return pockets[best]["center"], {
        "rank_E": rank_E[best],
        "rank_S": rank_S[best],
        "rank_C": rank_C[best],
        "combined": combined[best],
        "dist_to_af3_ligand": dists[best],
    }


def write_box(center, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"center_x = {center[0]:.3f}\n")
        f.write(f"center_y = {center[1]:.3f}\n")
        f.write(f"center_z = {center[2]:.3f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pocket_json", required=True)
    p.add_argument("--af3_ligand", required=True)
    p.add_argument("--out_box", required=True)
    p.add_argument("--lambda_", type=float, default=0.75)
    p.add_argument("--mu", type=float, default=1.0)
    args = p.parse_args()

    center, info = fused_box(args.pocket_json, args.af3_ligand,
                             lam=args.lambda_, mu=args.mu)
    write_box(center, args.out_box)
    print(f"✅ {args.out_box}")
    print(f"   rank_E={info['rank_E']}  rank_S={info['rank_S']}  "
          f"rank_C={info['rank_C']}  combined={info['combined']:.3f}")
    print(f"   dist_to_af3_ligand={info['dist_to_af3_ligand']:.2f} Å")


if __name__ == "__main__":
    main()