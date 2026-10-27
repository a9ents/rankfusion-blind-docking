# -*- coding: utf-8 -*-
"""
表格生成器（最终版）—— 读 v12.4 输出的 UTF-8-BOM CSV，生成两张表
=============================================================
表 1  整体分类统计
表 2  全部 case 明细（含 Rama / 各层 verdict）

输出：
  - 屏幕预览
  - table1_summary.csv
  - table2_detail.csv
  - table_report.txt
"""
import os
import csv
from collections import Counter
from datetime import datetime


OUT_DIR = r"E:\AF3环境\ALL-DATA\SMD_REPAIR_RESULTS"
CSV_IN  = os.path.join(OUT_DIR, "verification_summary.csv")
T1_CSV  = os.path.join(OUT_DIR, "table1_summary.csv")
T2_CSV  = os.path.join(OUT_DIR, "table2_detail.csv")
TXT     = os.path.join(OUT_DIR, "table_report.txt")


# ==================== 读取 ====================
def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ==================== 分类 ====================
def classify(r):
    case = (r.get("case") or "").strip()
    if not case:
        return "空行"

    if (r.get("error") or "").strip():
        return "报错"

    pf = (r.get("pre_filtered") or "").strip().lower()
    if pf in ("true", "1"):
        return "跳过_主链预过滤"

    gr = (r.get("geom_rejected") or "").strip().lower()
    if gr in ("true", "1"):
        return "跳过_几何拒绝"

    l1 = (r.get("L1_verdict") or "").strip()
    l2 = (r.get("L2_verdict") or "").strip()
    l3 = (r.get("L3_verdict") or "").strip()
    l4 = (r.get("L4_verdict") or "").strip()

    if not (l1 and l2 and l3 and l4):
        return "部分完成"

    if any("异常" in v for v in (l2, l3, l4)):
        return "完整完成_但异常"
    return "完整成功"


# ==================== 表 1 ====================
def build_table1(rows):
    cat = Counter(classify(r) for r in rows)
    n = len(rows)
    order = ["完整成功", "完整完成_但异常", "部分完成",
             "跳过_主链预过滤", "跳过_几何拒绝", "报错", "空行"]
    table = [("类别", "数量", "占比")]
    for k in order:
        if k in cat:
            table.append((k, cat[k], f"{cat[k]/n*100:.1f}%"))
    for k in cat:
        if k not in order:
            table.append((k, cat[k], f"{cat[k]/n*100:.1f}%"))
    table.append(("合计", n, "100.0%"))
    return table


# ==================== 表 2 ====================
def build_table2(rows):
    cols = [
        "case", "seed", "seed_probe", "seed_clash", "seed_init_clash",
        "pre_filtered", "filter_rama",
        "L1_drop", "L1_verdict",
        "L2_rama", "L2_clash", "L2_verdict",
        "L3_F_range_kJ", "L3_verdict",
        "L4_drop", "L4_verdict",
        "category", "error",
    ]
    out = []
    for r in rows:
        d = {c: (r.get(c) or "").strip() for c in cols if c != "category"}
        d["category"] = classify(r)
        out.append(d)

    order = {"完整成功": 0, "完整完成_但异常": 1, "部分完成": 2,
             "跳过_主链预过滤": 3, "跳过_几何拒绝": 4, "报错": 5, "空行": 6}
    out.sort(key=lambda d: (order.get(d["category"], 99), d["case"]))
    return cols, out


# ==================== 屏幕输出 ====================
def _disp_w(s):
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(s))

def _pad(s, w, align="left"):
    s = str(s)
    pad = w - _disp_w(s)
    if pad <= 0: return s
    return (" " * pad + s) if align == "right" else (s + " " * pad)

def print_table1(table):
    widths = [max(_disp_w(r[i]) for r in table) for i in range(3)]
    for i, row in enumerate(table):
        align = ["left", "right", "right"]
        cells = [_pad(row[j], widths[j], align[j] if i > 0 else "left")
                 for j in range(3)]
        print("  " + "  ".join(cells))
        if i == 0:
            print("  " + "  ".join("-" * widths[j] for j in range(3)))


# ==================== 主流程 ====================
def main():
    if not os.path.exists(CSV_IN):
        print(f"❌ 文件不存在: {CSV_IN}")
        return

    rows = read_csv(CSV_IN)
    print(f"\n[读取] {CSV_IN}")
    print(f"       有效行: {len(rows)}")
    if not rows:
        print("⚠ 无数据"); return

    # ---------- 表 1 ----------
    table1 = build_table1(rows)
    print(f"\n{'='*60}")
    print("表 1  整体分类统计")
    print(f"{'='*60}")
    print_table1(table1)

    # ---------- 表 2 ----------
    cols, detail = build_table2(rows)
    print(f"\n{'='*110}")
    print(f"表 2  全部 case 明细（共 {len(detail)} 个）")
    print(f"{'='*110}")

    # 只打印非"跳过_主链预过滤"的 case（真正值得看的）
    important = [d for d in detail if d["category"] != "跳过_主链预过滤"]
    print(f"\n  【完整跑完的 case：{len(important)} 个】")
    header = ["case", "category", "seed_probe", "filter_rama",
              "L1_verdict", "L2_verdict", "L3_verdict", "L4_verdict"]
    print("    " + "  ".join(f"{h:<14}" for h in header))
    for d in important:
        print("    " + "  ".join(
            f"{(d.get(h, '') or '')[:14]:<14}" for h in header
        ))

    # 预过滤跳过的只统计 Rama 分布
    skipped = [d for d in detail if d["category"] == "跳过_主链预过滤"]
    ramas = []
    for d in skipped:
        try: ramas.append(float(d.get("filter_rama", "") or 0))
        except: pass
    if ramas:
        import statistics
        print(f"\n  【{len(skipped)} 个主链预过滤跳过的 Rama 分布】")
        print(f"    min={min(ramas):.3f}  max={max(ramas):.3f}  "
              f"median={statistics.median(ramas):.3f}  "
              f"mean={statistics.mean(ramas):.3f}")
        buckets = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
        for lo, hi in buckets:
            c = sum(1 for x in ramas if lo <= x < hi)
            bar = "█" * c
            print(f"    [{lo:.2f}, {hi:.2f}): {c:>4}  {bar}")

    # ---------- 写文件 ----------
    with open(T1_CSV, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(table1)

    with open(T2_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(detail)

    with open(TXT, "w", encoding="utf-8-sig") as f:
        f.write(f"表格生成报告\n源: {CSV_IN}\n有效行={len(rows)}\n")
        f.write(f"生成: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write("=" * 60 + "\n表 1  整体分类统计\n" + "=" * 60 + "\n")
        for r in table1:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\n")
        f.write("\n" + "=" * 60 + "\n表 2  全部 case 明细\n" + "=" * 60 + "\n")
        f.write("\t".join(cols) + "\n")
        for d in detail:
            f.write("\t".join(str(d.get(c, "")) for c in cols) + "\n")

    print(f"\n输出:")
    print(f"  表 1 → {T1_CSV}")
    print(f"  表 2 → {T2_CSV}")
    print(f"  文本 → {TXT}")


if __name__ == "__main__":
    main()