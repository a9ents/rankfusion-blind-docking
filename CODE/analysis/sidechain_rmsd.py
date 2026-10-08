#!/usr/bin/env python3
"""
sidechain_rmsd.py — 口袋区域侧链 / 主链 RMSD 分析

对每个 case：
  1. 从实验结构找真值口袋残基（真值配体质心 8 Å 内）
  2. 在 AF3 对齐受体中按 Cα 最近邻匹配对应残基
  3. 算侧链重原子 RMSD 和主链重原子 RMSD
  4. 与盒子误差、ipTM、全局 RMSD 合并

输出：
  results/sidechain_rmsd.csv          逐 case
  results/sidechain_rmsd_summary.csv  分档统计
  results/sidechain_rmsd_report.txt   控制台报告

用法：
  python sidechain_rmsd.py
  python sidechain_rmsd.py --radius 8.0 --cases 6XG5_TOP,7LOE_Y84
"""
import argparse
import csv
from pathlib import Path
import numpy as np

try:
    import gemmi
except ImportError:
    print("需要 gemmi: pip install gemmi")
    raise


BACKBONE = {"N", "CA", "C", "O", "OXT"}


# =====================================================================
# 结构加载
# =====================================================================
def load_structure(path):
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()
    return st


def get_residue_list(st):
    """返回 [(chain_name, seq_num, res_name, res_obj), ...]"""
    out = []
    for chain in st[0]:
        for res in chain:
            out.append((chain.name, res.seqid.num, res.name, res))
    return out


def get_ca(res):
    for atom in res:
        if atom.name == "CA":
            return np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    return None


def get_backbone_coords(res):
    coords = []
    for atom in res:
        if atom.name in BACKBONE:
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.array(coords) if coords else None


def get_sidechain_coords(res):
    coords = []
    for atom in res:
        if atom.name not in BACKBONE:
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.array(coords) if coords else None


# =====================================================================
# 真值配体质心
# =====================================================================
def ligand_centroid(ligand_path):
    p = Path(ligand_path)
    suf = p.suffix.lower()

    if suf in (".sdf", ".mol"):
        try:
            from rdkit import Chem
            mol = Chem.MolFromMolFile(str(p), removeHs=True)
            if mol is None:
                return None
            conf = mol.GetConformer()
            coords = []
            for a in mol.GetAtoms():
                if a.GetAtomicNum() > 1:
                    pos = conf.GetAtomPosition(a.GetIdx())
                    coords.append([pos.x, pos.y, pos.z])
            if not coords:
                return None
            return np.mean(coords, axis=0)
        except Exception:
            return None

    if suf in (".pdb", ".ent"):
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
        if not coords:
            return None
        return np.mean(coords, axis=0)

    if suf == ".mol2":
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
        if not coords:
            return None
        return np.mean(coords, axis=0)

    return None


def pocket_residues_from_center(st, center, radius=8.0):
    c = gemmi.Position(*center)
    out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.pos.dist(c) < radius:
                    out.append((chain.name, res.seqid.num, res.name, res))
                    break
    return out


# =====================================================================
# 匹配 & RMSD
# =====================================================================
def match_residues(exp_res_list, af3_res_list, max_ca_dist=2.0):
    af3_index = []
    for (c, s, rn, res) in af3_res_list:
        ca = get_ca(res)
        if ca is not None:
            af3_index.append((c, s, rn, res, ca))
    if not af3_index:
        return []

    pairs = []
    for (ec, es, ern, eres) in exp_res_list:
        eca = get_ca(eres)
        if eca is None:
            continue

        best = None
        best_d = max_ca_dist
        for (ac, as_, arn, ares, aca) in af3_index:
            d = float(np.linalg.norm(eca - aca))
            if d < best_d:
                best_d = d
                best = ares
        if best is not None:
            pairs.append((eres, best))
    return pairs


def rmsd_from_pairs(pairs, coord_fn):
    diffs = []
    n_used = 0
    for (eres, ares) in pairs:
        e = coord_fn(eres)
        a = coord_fn(ares)
        if e is None or a is None:
            continue
        if len(e) != len(a):
            continue
        diffs.append(e - a)
        n_used += 1

    if not diffs:
        return None, 0
    all_d = np.vstack(diffs)
    rmsd = float(np.sqrt((all_d ** 2).sum(axis=1).mean()))
    return rmsd, n_used


# =====================================================================
# 单 case
# =====================================================================
def process_case(case, ligand_dir, exp_dir, af3_dir, radius=8.0):
    lig = None
    for name in ("ligand.sdf", "ligand.mol", "ligand.pdb", "ligand.mol2"):
        p = Path(ligand_dir) / case / name
        if p.is_file():
            lig = p
            break
    if lig is None:
        return None

    center = ligand_centroid(lig)
    if center is None:
        return None

    exp_rec = Path(exp_dir) / case / "receptor.pdb"
    af3_rec = Path(af3_dir) / case / "af3_top1" / "receptor.pdb"
    if not exp_rec.is_file() or not af3_rec.is_file():
        return None

    try:
        exp_st = load_structure(exp_rec)
        af3_st = load_structure(af3_rec)
    except Exception as e:
        print(f"  ⚠ {case}: 结构加载失败 {e}")
        return None

    pocket = pocket_residues_from_center(exp_st, center, radius)
    if not pocket:
        return None

    af3_res = get_residue_list(af3_st)
    pairs = match_residues(pocket, af3_res, max_ca_dist=2.0)
    if not pairs:
        return None

    sc_rmsd, n_sc = rmsd_from_pairs(pairs, get_sidechain_coords)
    bb_rmsd, n_bb = rmsd_from_pairs(pairs, get_backbone_coords)

    return {
        "case": case,
        "n_pocket_res": len(pocket),
        "n_matched": len(pairs),
        "n_sc_used": n_sc,
        "n_bb_used": n_bb,
        "sidechain_rmsd_A": round(sc_rmsd, 3) if sc_rmsd is not None else "",
        "backbone_rmsd_A": round(bb_rmsd, 3) if bb_rmsd is not None else "",
        "sc_over_bb": round(sc_rmsd / bb_rmsd, 3)
                       if (sc_rmsd and bb_rmsd and bb_rmsd > 0) else "",
    }


# =====================================================================
# 加载 CSV
# =====================================================================
def load_eval_map(path):
    """evaluation.csv，一行一个 case，包含各方法误差"""
    if not Path(path).is_file():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["case"]] = r
    return out


def load_af3_top1_map(path):
    """af3_model_summary.csv，只保留 model=af3_top1 的行"""
    if not Path(path).is_file():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("model") != "af3_top1":
                continue
            out[r["case"]] = r
    return out


def merge_all(rows, eval_csv, model_csv):
    ev = load_eval_map(eval_csv)
    md = load_af3_top1_map(model_csv)

    out = []
    for r in rows:
        case = r["case"]
        row = dict(r)

        # 盒子误差 & 命中
        if case in ev and "af3top1_rankfusion_err" in ev[case]:
            try:
                e = float(ev[case]["af3top1_rankfusion_err"])
                row["box_err_A"] = round(e, 3)
                row["hit_4A"] = 1 if e < 4.0 else 0
            except (ValueError, TypeError):
                row["box_err_A"] = ""
                row["hit_4A"] = ""
        else:
            row["box_err_A"] = ""
            row["hit_4A"] = ""

        # ipTM & 全局 RMSD
        if case in md:
            try:
                row["iptm"] = round(float(md[case]["iptm"]), 3)
            except (ValueError, TypeError, KeyError):
                row["iptm"] = ""
            try:
                row["global_rmsd_A"] = round(
                    float(md[case]["rmsd_to_exp"]), 3)
            except (ValueError, TypeError, KeyError):
                row["global_rmsd_A"] = ""
        else:
            row["iptm"] = ""
            row["global_rmsd_A"] = ""

        out.append(row)
    return out


# =====================================================================
# 分档统计
# =====================================================================
def bin_summary(rows, feature, bins, labels=None):
    ok = [r for r in rows
          if str(r.get(feature, "")) not in ("", "None")
          and str(r.get("hit_4A", "")) not in ("", "None")]
    if not ok:
        return []
    if labels is None:
        labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)]

    out = []
    for i, lab in enumerate(labels):
        lo, hi = bins[i], bins[i + 1]
        if i == len(labels) - 1:
            sub = [r for r in ok if lo <= float(r[feature]) <= hi]
        else:
            sub = [r for r in ok if lo <= float(r[feature]) < hi]
        if not sub:
            out.append({"bin": lab, "n": 0, "mean_feature": "",
                        "mean_box_err": "", "hit_4A": ""})
            continue
        feats = [float(r[feature]) for r in sub]
        errs = [float(r["box_err_A"]) for r in sub
                if str(r.get("box_err_A", "")) not in ("", "None")]
        hits = [int(r["hit_4A"]) for r in sub]
        out.append({
            "bin": lab,
            "n": len(sub),
            "mean_feature": round(float(np.mean(feats)), 3),
            "mean_box_err": round(float(np.mean(errs)), 3) if errs else "",
            "hit_4A": round(sum(hits) / len(hits), 4),
        })
    return out


# =====================================================================
# 主流程
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ligand_dir", default="ligands")
    p.add_argument("--exp_dir", default="exp_structures")
    p.add_argument("--af3_dir", default="results")
    p.add_argument("--eval_csv", default="results/evaluation.csv")
    p.add_argument("--model_csv", default="results/af3_model_summary.csv")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--cases", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        cases = sorted([d.name for d in Path(args.ligand_dir).iterdir()
                        if d.is_dir()])

    print(f"共 {len(cases)} 个 case，口袋半径 {args.radius} Å\n")

    rows = []
    for i, case in enumerate(cases, 1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{len(cases)}] {case}")
        r = process_case(case, args.ligand_dir, args.exp_dir,
                         args.af3_dir, args.radius)
        if r is not None:
            rows.append(r)

    if not rows:
        print("⚠ 无结果")
        return

    rows = merge_all(rows, args.eval_csv, args.model_csv)

    # 写逐 case
    cols = ["case", "n_pocket_res", "n_matched", "n_sc_used", "n_bb_used",
            "sidechain_rmsd_A", "backbone_rmsd_A", "sc_over_bb",
            "global_rmsd_A", "iptm", "box_err_A", "hit_4A"]
    for r in rows:
        for c in cols:
            r.setdefault(c, "")

    csv_path = out_dir / "sidechain_rmsd.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n✅ {csv_path}  ({len(rows)} cases)")

    # 分档
    bins = [0, 1, 2, 3, 4, 6, 8, 100]
    sum_sc = bin_summary(rows, "sidechain_rmsd_A", bins)
    sum_bb = bin_summary(rows, "backbone_rmsd_A", bins)
    sum_gl = bin_summary(rows, "global_rmsd_A", bins)

    sum_csv = out_dir / "sidechain_rmsd_summary.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "bin", "n", "mean_feature",
                    "mean_box_err", "hit_4A"])
        for tag, s in (("sidechain_rmsd", sum_sc),
                       ("backbone_rmsd", sum_bb),
                       ("global_rmsd", sum_gl)):
            for r in s:
                if r["n"] > 0:
                    w.writerow([tag, r["bin"], r["n"],
                                r["mean_feature"], r["mean_box_err"],
                                r["hit_4A"]])
    print(f"✅ {sum_csv}")

    # 报告
    report = []
    report.append("=" * 70)
    report.append("口袋区域侧链 / 主链 RMSD 分析")
    report.append(f"口袋半径: {args.radius} Å,  n_cases = {len(rows)}")
    report.append("=" * 70)

    valid = [r for r in rows
             if str(r.get("sidechain_rmsd_A", "")) not in ("", "None")
             and str(r.get("backbone_rmsd_A", "")) not in ("", "None")]
    if valid:
        sc = np.array([float(r["sidechain_rmsd_A"]) for r in valid])
        bb = np.array([float(r["backbone_rmsd_A"]) for r in valid])
        ratio = sc / np.maximum(bb, 1e-6)
        report.append(f"\n【口袋区域侧链 vs 主链 RMSD】 n={len(valid)}")
        report.append(f"  主链 RMSD  mean={bb.mean():.3f}  "
                      f"median={np.median(bb):.3f} Å")
        report.append(f"  侧链 RMSD  mean={sc.mean():.3f}  "
                      f"median={np.median(sc):.3f} Å")
        report.append(f"  侧链/主链  mean={ratio.mean():.3f}  "
                      f"median={np.median(ratio):.3f}")

    def print_bin(title, summary):
        report.append(f"\n【{title}分档 vs Rank Fusion】")
        report.append(f"  {'bin':<14}{'n':>6}{'mean':>10}"
                      f"{'box_err':>10}{'hit_4Å':>10}")
        for r in summary:
            if r["n"] == 0:
                continue
            report.append(f"  {r['bin']:<14}{r['n']:>6}"
                          f"{r['mean_feature']:>10.3f}"
                          f"{r['mean_box_err']:>10.3f}"
                          f"{r['hit_4A']*100:>9.2f}%")

    print_bin("侧链 RMSD (Å) ", sum_sc)
    print_bin("主链 RMSD (Å) ", sum_bb)
    print_bin("全局 RMSD (Å) ", sum_gl)

    # 相关性
    report.append("\n" + "=" * 70)
    report.append("相关性")
    report.append("=" * 70)
    try:
        from scipy.stats import spearmanr

        def corr(feat):
            pairs = [(r[feat], r["box_err_A"], r["hit_4A"])
                     for r in rows
                     if str(r.get(feat, "")) not in ("", "None")
                     and str(r.get("box_err_A", "")) not in ("", "None")]
            if len(pairs) < 5:
                return None
            a = np.array([float(p[0]) for p in pairs])
            b = np.array([float(p[1]) for p in pairs])
            h = np.array([int(p[2]) for p in pairs])
            rho1, p1 = spearmanr(a, b)
            rho2, p2 = spearmanr(a, h)
            return (rho1, p1, rho2, p2, len(pairs))

        c_sc = corr("sidechain_rmsd_A")
        if c_sc:
            report.append(f"\n侧链 RMSD vs 盒子误差:  "
                          f"ρ={c_sc[0]:+.3f}  p={c_sc[1]:.2e}  (n={c_sc[4]})")
            report.append(f"侧链 RMSD vs <4Å 命中:  "
                          f"ρ={c_sc[2]:+.3f}  p={c_sc[3]:.2e}")

        c_bb = corr("backbone_rmsd_A")
        if c_bb:
            report.append(f"主链 RMSD vs 盒子误差:  "
                          f"ρ={c_bb[0]:+.3f}  p={c_bb[1]:.2e}  (n={c_bb[4]})")
            report.append(f"主链 RMSD vs <4Å 命中:  "
                          f"ρ={c_bb[2]:+.3f}  p={c_bb[3]:.2e}")

        c_gl = corr("global_rmsd_A")
        if c_gl:
            report.append(f"全局 RMSD vs 盒子误差:  "
                          f"ρ={c_gl[0]:+.3f}  p={c_gl[1]:.2e}  (n={c_gl[4]})")
            report.append(f"全局 RMSD vs <4Å 命中:  "
                          f"ρ={c_gl[2]:+.3f}  p={c_gl[3]:.2e}")
    except ImportError:
        report.append("  (需要 scipy)")

    # 匹配失败 vs 成功的成功率对比
    matched = [r for r in rows
               if str(r.get("sidechain_rmsd_A", "")) not in ("", "None")
               and str(r.get("hit_4A", "")) not in ("", "None")]
    unmatched = [r for r in rows
                 if str(r.get("sidechain_rmsd_A", "")) in ("", "None")
                 and str(r.get("hit_4A", "")) not in ("", "None")]
    report.append("\n" + "=" * 70)
    report.append("匹配成功 vs 失败 case 的成功率对比")
    report.append("=" * 70)
    for tag, sub in (("matched", matched), ("unmatched", unmatched)):
        if not sub:
            continue
        hits = [int(r["hit_4A"]) for r in sub]
        errs = [float(r["box_err_A"]) for r in sub
                if str(r.get("box_err_A", "")) not in ("", "None")]
        report.append(f"  {tag:<10} n={len(sub):<4}  "
                      f"hit_4Å={sum(hits)/len(hits)*100:>6.2f}%  "
                      f"mean_box_err={np.mean(errs):>7.2f} Å")

    text = "\n".join(report)
    print("\n" + text)
    rp = out_dir / "sidechain_rmsd_report.txt"
    with open(rp, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n✅ {rp}")


if __name__ == "__main__":
    main()