#!/usr/bin/env python3
"""af3_p2rank_detection.py — AF3 结构上 P2Rank 检测能力对照。"""
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
import af3_common as ac

# ============ 绝对路径配置 ============
AF3_ROOT = Path(r"E:\AF3环境\AF3_workflow")
EXP_REC_ROOT = Path(r"E:\AF3环境\exp_structures")
LIGAND_ROOT = Path(r"E:\AF3环境\ALL-DATA\DATA\ligands")
OUT_DIR = Path(r"E:\AF3环境\ALL-DATA\results")
OUT_CSV = OUT_DIR / "af3_p2rank_detection.csv"

P2RANK_DIR = Path(r"E:\AF3环境\p2rank_2.6-alpha")
JAVA_BIN = "java"
WORK_ROOT = Path(r"E:\AF3环境\p2rank_af3_work")

JAVA_OPTS = [
    "-Xmx2048m",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/jdk.internal.misc=ALL-UNNAMED",
]


def build_classpath():
    return f"{P2RANK_DIR / 'bin' / 'p2rank.jar'};{P2RANK_DIR / 'bin' / 'lib' / '*'}"


def find_af3_cif(case):
    """找 AF3 CIF：优先 SEED-1_SAMPLE-0 下的 *.cif，其次任何 SEED，最后 MODEL.cif。"""
    case_dir = AF3_ROOT / case
    if not case_dir.is_dir():
        return None
    s0 = case_dir / "SEED-1_SAMPLE-0"
    if s0.is_dir():
        cifs = sorted(s0.glob("*.cif"))
        if cifs:
            return cifs[0]
    for sd in sorted(case_dir.glob("SEED-1_SAMPLE-*")):
        cifs = sorted(sd.glob("*.cif"))
        if cifs:
            return cifs[0]
    model = case_dir / f"{case}_MODEL.cif"
    if model.is_file():
        return model
    return None


def find_ligand_with_centroid(case):
    """依次尝试各配体文件，返回 (path, centroid) 或 (None, None)。"""
    ld = LIGAND_ROOT / case
    if not ld.is_dir():
        return None, None
    for nm in ("ligand.sdf", "ligand.mol", "ligand.pdb", "ligand.mol2"):
        p = ld / nm
        if p.is_file():
            try:
                c = ac.ligand_centroid(p)
            except Exception:
                c = None
            if c is not None:
                return p, c
    return None, None


def run_p2rank(pdb_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=180)
    except subprocess.TimeoutExpired:
        return None
    stem = Path(pdb_path).stem
    for csv_file in out_dir.rglob(f"{stem}.pdb_predictions.csv"):
        return csv_file
    return None


def parse_predictions(csv_path):
    pockets = []
    if csv_path is None or not Path(csv_path).is_file():
        return pockets
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v.strip() if isinstance(v, str) else v)
                   for k, v in row.items()}
            try:
                c = np.array([float(row["center_x"]),
                              float(row["center_y"]),
                              float(row["center_z"])])
                pockets.append({
                    "rank": int(row.get("rank", 999)),
                    "score": float(row.get("score", 0)),
                    "center": c,
                })
            except (KeyError, ValueError):
                continue
    pockets.sort(key=lambda p: p["rank"])
    return pockets


def process_one(case):
    # 1. AF3 CIF
    cif = find_af3_cif(case)
    if cif is None:
        return {"case": case, "status": "no_cif"}

    # 2. 实验受体
    exp_rec = EXP_REC_ROOT / case / "receptor.pdb"
    if not exp_rec.is_file():
        return {"case": case, "status": "no_exp_rec"}

    # 3. 真值配体（自动尝试多格式）
    lig, true_c = find_ligand_with_centroid(case)
    if true_c is None:
        return {"case": case, "status": "no_centroid"}

    # 4. 提取受体 + 对齐
    work = WORK_ROOT / case
    work.mkdir(parents=True, exist_ok=True)
    raw_rec = work / "af3_rec_raw.pdb"
    ok_r, _ = ac.split_af3_cif(cif, raw_rec, None)
    if not ok_r:
        return {"case": case, "status": "split_fail"}

    aligned = work / "af3_rec_aligned.pdb"
    rmsd, tm = ac.run_usalign(raw_rec, exp_rec, aligned)
    if rmsd is None or not aligned.is_file():
        return {"case": case, "status": "align_fail"}

    # 5. P2Rank
    p2rank_out = work / "p2rank"
    if p2rank_out.exists():
        shutil.rmtree(p2rank_out)
    pred_csv = run_p2rank(str(aligned), str(p2rank_out))
    pockets = parse_predictions(pred_csv)
    if not pockets:
        return {"case": case, "status": "no_pocket",
                "rmsd": round(rmsd, 3),
                "tm": round(tm, 3) if tm is not None else ""}

    # 6. 分析
    dists = [float(np.linalg.norm(p["center"] - true_c)) for p in pockets]
    best_idx = int(np.argmin(dists))
    best_d = dists[best_idx]
    top20 = pockets[:20]
    top20_dists = [float(np.linalg.norm(p["center"] - true_c)) for p in top20]
    best_in_top20 = min(top20_dists)

    return {
        "case": case,
        "status": "ok",
        "rmsd": round(rmsd, 3),
        "tm": round(tm, 3) if tm is not None else "",
        "n_pockets": len(pockets),
        "min_dist": round(best_d, 3),
        "rank_of_best": best_idx + 1,
        "top20_oracle_dist": round(best_in_top20, 3),
        "hit_4A": 1 if best_d < 4 else 0,
        "hit_8A": 1 if best_d < 8 else 0,
        "oracle_4A": 1 if best_in_top20 < 4 else 0,
    }


def main():
    print(f"[info] AF3_ROOT      = {AF3_ROOT}")
    print(f"[info] EXP_REC_ROOT  = {EXP_REC_ROOT}")
    print(f"[info] LIGAND_ROOT   = {LIGAND_ROOT}")
    print(f"[info] OUT_CSV       = {OUT_CSV}")

    if not AF3_ROOT.is_dir():
        sys.exit(f"ERROR: {AF3_ROOT} not found")
    if not EXP_REC_ROOT.is_dir():
        sys.exit(f"ERROR: {EXP_REC_ROOT} not found")
    if not LIGAND_ROOT.is_dir():
        sys.exit(f"ERROR: {LIGAND_ROOT} not found")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    cases = sorted([d.name for d in AF3_ROOT.iterdir() if d.is_dir()])
    print(f"\n发现 {len(cases)} 个 AF3 case")

    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(process_one, c): c for c in cases}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                r = fut.result()
            except Exception as e:
                r = {"case": futures[fut], "status": f"crash_{str(e)[:40]}"}
            if r:
                rows.append(r)
            if done % 20 == 0 or done == len(cases):
                n_ok = sum(1 for x in rows if x.get("status") == "ok")
                print(f"  进度 {done}/{len(cases)}  成功 {n_ok}")

    keys = ["case", "status", "rmsd", "tm", "n_pockets", "min_dist",
            "rank_of_best", "top20_oracle_dist", "hit_4A", "hit_8A", "oracle_4A"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {OUT_CSV}  ({len(rows)} 行)")

    # 报告
    ok = [r for r in rows if r.get("status") == "ok"]
    L = []
    L.append("=" * 72)
    L.append("AF3 结构上 P2Rank 检测能力对照")
    L.append("=" * 72)
    L.append(f"\n总 case: {len(rows)}")
    L.append(f"成功:   {len(ok)}")

    if len(rows) - len(ok) > 0:
        from collections import Counter
        c = Counter(r["status"] for r in rows if r.get("status") != "ok")
        L.append("失败原因:")
        for k, v in c.most_common():
            L.append(f"  {k}: {v}")

    if ok:
        n = len(ok)
        dists = np.array([r["min_dist"] for r in ok])
        h4 = np.array([r["hit_4A"] for r in ok])
        h8 = np.array([r["hit_8A"] for r in ok])
        o4 = np.array([r["oracle_4A"] for r in ok])
        np_ = np.array([r["n_pockets"] for r in ok])

        L.append(f"\nP2Rank 检测到的 pocket 数: mean={np_.mean():.1f}, "
                 f"median={np.median(np_):.0f}")
        L.append(f"\n最近 pocket 到真值中心距离 (Å):")
        L.append(f"  mean   = {dists.mean():.2f}")
        L.append(f"  median = {np.median(dists):.2f}")
        L.append(f"  Q1–Q3  = {np.percentile(dists,25):.2f} – {np.percentile(dists,75):.2f}")
        L.append(f"\n命中率:")
        L.append(f"  最近 pocket < 4 Å:        {h4.sum()} ({h4.mean()*100:.2f}%)")
        L.append(f"  最近 pocket < 8 Å:        {h8.sum()} ({h8.mean()*100:.2f}%)")
        L.append(f"  前 20 pocket oracle < 4 Å: {o4.sum()} ({o4.mean()*100:.2f}%)")

        L.append(f"\n对比 fpocket（论文 5.4 节）:")
        L.append(f"  fpocket 最近 pocket 中位距离: 24.34 Å")
        L.append(f"  fpocket 最近 pocket < 4 Å:    0.9%")

        L.append("\n" + "=" * 72)
        L.append("结论")
        L.append("=" * 72)
        if h4.mean() < 0.05:
            L.append("\n  P2Rank 在 AF3 结构上同样检测不到真值口袋，")
            L.append("  证明「检测失效」不是 fpocket 特有，而是结构层面的问题。")
        else:
            L.append(f"\n  P2Rank 在 AF3 结构上仍有 {h4.mean()*100:.1f}% 的 case 检测到真值口袋，")
            L.append("  说明不同检测器对结构误差的鲁棒性存在差异。")

    text = "\n".join(L)
    print("\n" + text)
    report_path = OUT_DIR / "af3_p2rank_detection_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[OK] {report_path}")


if __name__ == "__main__":
    main()