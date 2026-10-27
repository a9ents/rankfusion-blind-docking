#!/usr/bin/env python3
"""
docking_software.py — 大分子-小分子盲对接软件（完整版）

流程：
  1. 选择大分子文件夹（*.pdb）
  2. 选择小分子父文件夹（每个 case 一个子目录）
  3. 选择 Box 策略：
       - fpocket + Rank Fusion（默认，需 WSL + fpocket）
       - P2Rank + Rank Fusion（需 P2Rank）
       - 手动指定 box（所有 case 共用一个 .box）
       - 盲对接（整蛋白 bbox）
  4. 选择对接精度：快速 / 标准 / 深度 / 自定义
  5. 一键运行，实时进度 + 日志

所有路径自适应：环境变量 → 项目 tools/ → PATH
任何电脑拷贝整个文件夹即可运行，无需改代码。

CLI 模式:
  python docking_software.py --cli \\
      --receptor_dir exp_structures --ligand_parent ligands \\
      --out_dir results --strategy fpocket_rf
"""
import os
import sys
import shutil
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


# =====================================================================
# 路径自适应导入
# =====================================================================
_ROOT = Path(__file__).resolve().parent
for _sub in ("pipeline", "src", "analysis"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import auto_pipeline as ap                    # noqa: E402
import run_p2rank_rankfusion as prf           # noqa: E402
import run_p2rank_baseline as prb             # noqa: E402


# =====================================================================
# 全局常量
# =====================================================================
BOX_STRATEGIES = [
    ("fpocket_rf", "fpocket + Rank Fusion（推荐）"),
    ("p2rank_rf",  "P2Rank + Rank Fusion"),
    ("manual",     "手动指定 box"),
    ("blind",      "盲对接（整蛋白 bbox）"),
]

PRESETS = {
    "fast":     {"exh": 8,  "modes": 5,  "label": "快速（exh=8, modes=5）"},
    "standard": {"exh": 32, "modes": 20, "label": "标准（exh=32, modes=20）"},
    "deep":     {"exh": 64, "modes": 40, "label": "深度（exh=64, modes=40）"},
    "custom":   {"exh": 32, "modes": 20, "label": "自定义"},
}

WORK_ROOT = _ROOT / "work"


# =====================================================================
# 工具路径解析（统一入口）
# =====================================================================
def resolve_vina():
    """Vina 路径：环境变量 → PATH → tools/vina/vina.exe"""
    v = os.environ.get("VINA_BIN")
    if v and Path(v).is_file():
        return v
    p = ap._which("vina")
    if p:
        return p
    cand = _ROOT / "tools" / "vina" / "vina.exe"
    return str(cand) if cand.is_file() else None


def resolve_p2rank_dir():
    """P2Rank 安装目录：环境变量 → tools/p2rank_2.6-alpha"""
    v = os.environ.get("P2RANK_DIR")
    if v and (Path(v) / "bin" / "p2rank.jar").is_file():
        return Path(v)
    cand = _ROOT / "tools" / "p2rank_2.6-alpha"
    if (cand / "bin" / "p2rank.jar").is_file():
        return cand
    return None


# =====================================================================
# 环境自检
# =====================================================================
def check_environment(log):
    """返回 True 表示必需工具齐全。"""
    ok = True

    vina = resolve_vina()
    if vina:
        log(f"  ✓ vina: {vina}")
    else:
        log("  ✗ vina 未找到（必需）→ 放到 tools/vina/ 或加到 PATH")
        ok = False

    obabel = ap._which("obabel")
    if obabel:
        log(f"  ✓ obabel: {obabel}")
    else:
        log("  ✗ obabel 未找到（必需）→ 加到 PATH")
        ok = False

    meeko = ap._which("mk_prepare_ligand")
    if meeko:
        log(f"  ✓ mk_prepare_ligand: {meeko}")
    else:
        log("  · mk_prepare_ligand 未找到（可选，回退 obabel）")

    p2rank = resolve_p2rank_dir()
    if p2rank:
        log(f"  ✓ P2Rank: {p2rank}")
    else:
        log("  · P2Rank 未找到（仅 p2rank_rf 策略需要）")

    return ok


# =====================================================================
# 单 case 处理
# =====================================================================
def process_one_case(case, receptor_pdb, ligand_dir, out_root,
                     strategy, cfg, log):
    case_out = Path(out_root) / case
    case_out.mkdir(parents=True, exist_ok=True)

    log(f"\n[{case}] 大分子: {receptor_pdb.name}")
    log(f"[{case}] 小分子: {ligand_dir}")

    # ---- 受体 pdbqt ----
    rec_pdbqt = case_out / "receptor.pdbqt"
    if not rec_pdbqt.is_file() or rec_pdbqt.stat().st_size < 100:
        log(f"[{case}]   → 转换受体 pdbqt")
        if not ap.pdb_to_pdbqt_receptor(str(receptor_pdb), str(rec_pdbqt)):
            log(f"[{case}]   ✗ 受体转换失败")
            return False

    # ---- 收集配体 ----
    ligs = sorted([
        p for p in Path(ligand_dir).iterdir()
        if p.suffix.lower() in (".pdb", ".sdf", ".mol", ".mol2", ".pdbqt")
        and p.is_file()
    ])
    if not ligs:
        log(f"[{case}]   ✗ 配体目录无有效文件")
        return False

    lig_work = case_out / "ligands_pdbqt"
    lig_work.mkdir(exist_ok=True)
    lig_pdbqts = []
    for lig in ligs:
        out_pqt = lig_work / f"{lig.stem}.pdbqt"
        if not out_pqt.is_file() or out_pqt.stat().st_size < 100:
            if not ap.pdb_to_pdbqt_ligand(str(lig), str(out_pqt)):
                continue
        lig_pdbqts.append(out_pqt)
    if not lig_pdbqts:
        log(f"[{case}]   ✗ 无配体成功转 pdbqt")
        return False
    log(f"[{case}]   → {len(lig_pdbqts)} 个配体就绪")

    # ---- 按策略生成 box ----
    box_file = case_out / "receptor.box"
    final_center = None
    final_size = None

    if strategy == "blind":
        log(f"[{case}]   → 盲对接 bbox")
        center, size = ap.compute_bounding_box(
            str(rec_pdbqt), ap.CFG["blind_padding"],
            ap.CFG["blind_max_size"])
        if center is None:
            log(f"[{case}]   ✗ bbox 计算失败")
            return False
        final_center = center
        final_size = size
        with open(box_file, "w") as f:
            f.write(f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f} "
                    f"{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}\n")

    elif strategy == "manual":
        manual_src = cfg.get("manual_box")
        if not manual_src or not Path(manual_src).is_file():
            log(f"[{case}]   ✗ 手动 box 未指定")
            return False
        with open(manual_src) as f:
            parts = f.read().split()
        final_center = tuple(map(float, parts[:3]))
        final_size = tuple(map(float, parts[3:6]))
        with open(box_file, "w") as f:
            f.write(f"{final_center[0]:.3f} {final_center[1]:.3f} "
                    f"{final_center[2]:.3f} {final_size[0]:.3f} "
                    f"{final_size[1]:.3f} {final_size[2]:.3f}\n")
        log(f"[{case}]   → 使用手动 box")

    elif strategy == "fpocket_rf":
        log(f"[{case}]   → fpocket + Rank Fusion")
        ap.CFG["blind_mode"] = "pocket_guided"
        ap.CFG["pg_rank_lambda"] = cfg.get("lambda", 0.75)
        ap.CFG["pg_quick_exh"] = 4
        ap.CFG["pg_quick_modes"] = 3
        ap.CFG["pg_max_pockets"] = cfg.get("max_pockets", 20)
        ap.CFG["pg_box_size"] = cfg.get("box_size", 25.0)
        try:
            ok, _ = ap.run_pocket_guided(
                str(receptor_pdb), str(ligand_dir), str(case_out))
        except Exception as e:
            log(f"[{case}]   ✗ fpocket 异常: {e}")
            return False
        if not ok or not box_file.is_file():
            log(f"[{case}]   ✗ fpocket 未生成 box")
            return False
        with open(box_file) as f:
            parts = f.read().split()
        final_center = tuple(map(float, parts[:3]))
        final_size = tuple(map(float, parts[3:6]))

    elif strategy == "p2rank_rf":
        log(f"[{case}]   → P2Rank + Rank Fusion")
        ok, center, size = _run_p2rank_guided(
            receptor_pdb, ligand_dir, case_out, cfg, log)
        if not ok:
            return False
        final_center = center
        final_size = size

    else:
        log(f"[{case}]   ✗ 未知策略: {strategy}")
        return False

    # ---- 精修对接 ----
    log(f"[{case}]   → 精修 exh={cfg['exh']} modes={cfg['modes']}")
    fine_dir = case_out / "docking_results"
    fine_dir.mkdir(exist_ok=True)
    ok_n = 0
    for i, lig_pqt in enumerate(lig_pdbqts, 1):
        out_pqt = fine_dir / f"docking_{lig_pqt.stem}.pdbqt"
        if out_pqt.is_file() and out_pqt.stat().st_size > 100:
            ok_n += 1
            continue
        log(f"[{case}]     ({i}/{len(lig_pdbqts)}) {lig_pqt.stem}")
        if ap.run_vina(str(rec_pdbqt), str(lig_pqt), str(out_pqt),
                       final_center, final_size,
                       cfg["exh"], cfg["modes"], 42):
            ok_n += 1
    log(f"[{case}]   ✓ 完成 {ok_n}/{len(lig_pdbqts)}")
    return ok_n > 0


# =====================================================================
# P2Rank 引导子流程
# =====================================================================
def _run_p2rank_guided(receptor_pdb, ligand_dir, case_out, cfg, log):
    """返回 (ok, center, size)。"""
    receptor_pdb = Path(receptor_pdb)
    ligand_dir = Path(ligand_dir)
    case_out = Path(case_out)
    case = receptor_pdb.stem

    # 1. 检查 P2Rank
    p2rank_dir = resolve_p2rank_dir()
    if p2rank_dir is None:
        log(f"[{case}]   ✗ P2Rank 未找到")
        log(f"[{case}]     请放到 {_ROOT / 'tools' / 'p2rank_2.6-alpha'}")
        log(f"[{case}]     或设置环境变量 P2RANK_DIR")
        return False, None, None

    # 2. 检查配体目录里有 .pdb
    pdb_ligs = list(ligand_dir.glob("*.pdb"))
    if not pdb_ligs:
        log(f"[{case}]   ⚠ 配体目录无 .pdb 文件，尝试从 sdf/mol 转换…")
        sdf_ligs = list(ligand_dir.glob("*.sdf")) + \
                   list(ligand_dir.glob("*.mol"))
        if sdf_ligs and ap.OBABEL_BIN:
            probe_pdb = ligand_dir / "_probe_from_sdf.pdb"
            import subprocess
            try:
                subprocess.run(
                    [ap.OBABEL_BIN, str(sdf_ligs[0]), "-O",
                     str(probe_pdb)], capture_output=True, timeout=60)
            except Exception:
                pass
            if probe_pdb.is_file():
                pdb_ligs = [probe_pdb]
        if not pdb_ligs:
            log(f"[{case}]   ✗ 无法获得 .pdb 配体，P2Rank 引导失败")
            return False, None, None

    # 3. 设置 prf 路径变量
    p2rank_out = WORK_ROOT / "p2rank_out"
    p2rank_out.mkdir(parents=True, exist_ok=True)
    prf.P2RANK_OUT_ROOT = p2rank_out
    prf.WORK_ROOT = WORK_ROOT / "p2rank_rf"
    prf.WORK_ROOT.mkdir(parents=True, exist_ok=True)

    # 4. 未跑过 P2Rank 则先跑
    has_pred = any(p2rank_out.rglob(f"{case}.pdb_predictions.csv"))
    if not has_pred:
        log(f"[{case}]   → 先运行 P2Rank 检测")
        prb.P2RANK_DIR = p2rank_dir
        prb.WORK_ROOT = WORK_ROOT / "p2rank_work"
        prb.WORK_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            _, info = prb.run_one(
                (case, str(receptor_pdb), str(p2rank_out)))
        except Exception as e:
            log(f"[{case}]   ✗ P2Rank 运行异常: {e}")
            return False, None, None
        if info is None:
            log(f"[{case}]   ✗ P2Rank 未产出预测")
            return False, None, None

    # 5. 构造适配 run_p2rank_rankfusion 的目录结构
    #    期望 exp_dir/case/receptor.pdb
    tmp_exp = case_out / "_tmp_exp"
    tmp_case = tmp_exp / case
    tmp_case.mkdir(parents=True, exist_ok=True)
    tmp_rec = tmp_case / "receptor.pdb"
    if not tmp_rec.is_file():
        shutil.copy2(receptor_pdb, tmp_rec)

    # 期望 ligand_parent/case/*.pdb
    if ligand_dir.name == case:
        # 配体目录本身就是 case 目录（标准布局）
        ligand_parent_arg = ligand_dir.parent
    else:
        # 名字不同，构造临时结构
        tmp_lig_parent = case_out / "_tmp_lig"
        tmp_lig_case = tmp_lig_parent / case
        tmp_lig_case.mkdir(parents=True, exist_ok=True)
        for lig in ligand_dir.iterdir():
            if lig.is_file() and lig.suffix.lower() == ".pdb":
                dst = tmp_lig_case / lig.name
                if not dst.is_file():
                    shutil.copy2(lig, dst)
        ligand_parent_arg = tmp_lig_parent

    # 6. 调用 Rank Fusion
    try:
        _, center, info = prf.process_one(
            case, str(tmp_exp), str(ligand_parent_arg), str(case_out))
    except Exception as e:
        log(f"[{case}]   ✗ P2Rank 引导异常: {e}")
        return False, None, None

    if center is None:
        log(f"[{case}]   ✗ P2Rank 引导失败: {info}")
        return False, None, None

    # 7. 复制 box
    src_box = case_out / f"p2rank_rf_{case}.box"
    dst_box = case_out / "receptor.box"
    box_size = cfg.get("box_size", 25.0)
    if src_box.is_file():
        shutil.copy(src_box, dst_box)
    else:
        with open(dst_box, "w") as f:
            f.write(f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f} "
                    f"{box_size:.3f} {box_size:.3f} {box_size:.3f}\n")
    return True, tuple(center), (box_size, box_size, box_size)


# =====================================================================
# 批量运行线程
# =====================================================================
def run_all_thread(receptor_dir, ligand_parent, out_root, strategy,
                   cfg, log, progress_cb, stop_event):
    rec_dir = Path(receptor_dir)
    lig_parent = Path(ligand_parent)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pdbs = sorted(p for p in rec_dir.glob("*.pdb") if p.is_file())
    if not pdbs:
        log(f"❌ {rec_dir} 下无 *.pdb")
        return

    log(f"\n发现 {len(pdbs)} 个大分子")
    ok = fail = 0
    for i, rec in enumerate(pdbs, 1):
        if stop_event.is_set():
            log("\n⏹ 用户中断")
            break
        case = rec.stem
        lig_dir = lig_parent / case
        if not lig_dir.is_dir():
            log(f"[{i}/{len(pdbs)}] {case}   ⚠ 无配体目录，跳过")
            fail += 1
            progress_cb(i, len(pdbs), case)
            continue
        progress_cb(i - 1, len(pdbs), case)
        try:
            if process_one_case(case, rec, lig_dir, out_root,
                                strategy, cfg, log):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            log(f"[{i}/{len(pdbs)}] {case}   ❌ 异常: {e}")
            traceback.print_exc()
            fail += 1
        progress_cb(i, len(pdbs), case)

    log(f"\n{'=' * 60}")
    log(f"✅ 完成: 成功 {ok} / 失败 {fail} / 总 {len(pdbs)}")
    log(f"📁 输出: {out_root.resolve()}")
    log(f"{'=' * 60}")


# =====================================================================
# GUI
# =====================================================================
class DockingApp:
    def __init__(self, root):
        self.root = root
        root.title("大分子-小分子对接软件")
        root.geometry("980x780")
        self._stop_event = threading.Event()
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        # ===== 顶部路径 =====
        top = ttk.LabelFrame(self.root, text="1. 数据路径", padding=10)
        top.pack(fill="x", padx=10, pady=6)

        ttk.Label(top, text="大分子文件夹:").grid(
            row=0, column=0, sticky="w")
        self.rec_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.rec_var, width=70).grid(
            row=0, column=1, padx=6)
        ttk.Button(top, text="选择…",
                   command=self._pick_rec).grid(row=0, column=2)

        ttk.Label(top, text="小分子父文件夹:").grid(
            row=1, column=0, sticky="w", pady=4)
        self.lig_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.lig_var, width=70).grid(
            row=1, column=1, padx=6)
        ttk.Button(top, text="选择…",
                   command=self._pick_lig).grid(row=1, column=2)

        ttk.Label(top, text="输出目录:").grid(
            row=2, column=0, sticky="w", pady=4)
        self.out_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.out_var, width=70).grid(
            row=2, column=1, padx=6)
        ttk.Button(top, text="选择…",
                   command=self._pick_out).grid(row=2, column=2)

        # ===== Box 策略 =====
        mid = ttk.LabelFrame(self.root, text="2. Box 选择策略", padding=10)
        mid.pack(fill="x", padx=10, pady=6)

        self.strategy_var = tk.StringVar(value="fpocket_rf")
        for key, label in BOX_STRATEGIES:
            ttk.Radiobutton(
                mid, text=label, variable=self.strategy_var,
                value=key, command=self._on_strategy_change
            ).pack(anchor="w")

        manual_row = ttk.Frame(mid)
        manual_row.pack(fill="x", pady=(6, 0))
        ttk.Label(manual_row, text="  手动 box 文件:").pack(side="left")
        self.manual_box_var = tk.StringVar()
        self.manual_entry = ttk.Entry(
            manual_row, textvariable=self.manual_box_var, width=55)
        self.manual_entry.pack(side="left", padx=6)
        self.manual_btn = ttk.Button(
            manual_row, text="选择…", command=self._pick_manual_box)
        self.manual_btn.pack(side="left")
        self._manual_widgets = [self.manual_entry, self.manual_btn]
        self._on_strategy_change()

        # ===== 参数 =====
        cfg = ttk.LabelFrame(self.root, text="3. 对接参数", padding=10)
        cfg.pack(fill="x", padx=10, pady=6)

        ttk.Label(cfg, text="预设:").grid(row=0, column=0, sticky="w")
        self.preset_var = tk.StringVar(value=PRESETS["standard"]["label"])
        preset_box = ttk.Combobox(
            cfg, textvariable=self.preset_var, state="readonly",
            values=[v["label"] for v in PRESETS.values()], width=26)
        preset_box.grid(row=0, column=1, sticky="w", padx=6)
        preset_box.bind("<<ComboboxSelected>>", self._on_preset_change)

        ttk.Label(cfg, text="exhaustiveness:").grid(
            row=0, column=2, padx=(20, 4), sticky="w")
        self.exh_var = tk.IntVar(value=32)
        ttk.Spinbox(cfg, from_=1, to=128, textvariable=self.exh_var,
                    width=8).grid(row=0, column=3)

        ttk.Label(cfg, text="num_modes:").grid(
            row=0, column=4, padx=(20, 4), sticky="w")
        self.modes_var = tk.IntVar(value=20)
        ttk.Spinbox(cfg, from_=1, to=100, textvariable=self.modes_var,
                    width=8).grid(row=0, column=5)

        ttk.Label(cfg, text="Rank Fusion λ:").grid(
            row=1, column=0, sticky="w", pady=4)
        self.lambda_var = tk.DoubleVar(value=0.75)
        ttk.Spinbox(cfg, from_=0.0, to=10.0, increment=0.05,
                    textvariable=self.lambda_var,
                    width=10).grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(cfg, text="最大口袋数:").grid(
            row=1, column=2, padx=(20, 4), sticky="w")
        self.maxp_var = tk.IntVar(value=20)
        ttk.Spinbox(cfg, from_=1, to=100,
                    textvariable=self.maxp_var, width=8).grid(
            row=1, column=3)

        ttk.Label(cfg, text="box 边长 (Å):").grid(
            row=1, column=4, padx=(20, 4), sticky="w")
        self.boxsz_var = tk.DoubleVar(value=25.0)
        ttk.Spinbox(cfg, from_=10.0, to=60.0, increment=1.0,
                    textvariable=self.boxsz_var, width=8).grid(
            row=1, column=5)

        # ===== 控制按钮 =====
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=10, pady=(4, 6))
        self.run_btn = ttk.Button(ctrl, text="▶ 开始运行",
                                  command=self._start)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(ctrl, text="⏹ 停止",
                                   command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)

        ttk.Label(ctrl, text="进度:").pack(side="left", padx=(20, 4))
        self.pbar = ttk.Progressbar(ctrl, length=420)
        self.pbar.pack(side="left")
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(ctrl, textvariable=self.status_var).pack(
            side="left", padx=10)

        # ===== 日志 =====
        logf = ttk.LabelFrame(self.root, text="运行日志", padding=6)
        logf.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_txt = scrolledtext.ScrolledText(
            logf, height=20, font=("Consolas", 9), wrap="word")
        self.log_txt.pack(fill="both", expand=True)

    # -------- 路径选择 --------
    def _pick_rec(self):
        d = filedialog.askdirectory(title="选择大分子文件夹")
        if d:
            self.rec_var.set(d)
            if not self.out_var.get():
                self.out_var.set(str(Path(d).parent / "docking_out"))

    def _pick_lig(self):
        d = filedialog.askdirectory(title="选择小分子父文件夹")
        if d:
            self.lig_var.set(d)

    def _pick_out(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.out_var.set(d)

    def _pick_manual_box(self):
        f = filedialog.askopenfilename(
            title="选择 box 文件",
            filetypes=[("Box files", "*.box"), ("All", "*.*")])
        if f:
            self.manual_box_var.set(f)

    def _on_strategy_change(self):
        state = "normal" if self.strategy_var.get() == "manual" \
                else "disabled"
        for w in self._manual_widgets:
            w.configure(state=state)

    def _on_preset_change(self, _e=None):
        label = self.preset_var.get()
        for key, p in PRESETS.items():
            if p["label"] == label and key != "custom":
                self.exh_var.set(p["exh"])
                self.modes_var.set(p["modes"])
                return

    # -------- 运行控制 --------
    def _log(self, msg):
        self.log_txt.insert("end", msg + "\n")
        self.log_txt.see("end")

    def _log_safe(self, msg):
        self.root.after(0, self._log, msg)

    def _set_progress(self, done, total, case):
        self.pbar["maximum"] = total
        self.pbar["value"] = done
        self.status_var.set(f"{done}/{total}  {case}")

    def _set_progress_safe(self, done, total, case):
        self.root.after(0, self._set_progress, done, total, case)

    def _start(self):
        rec = self.rec_var.get().strip()
        lig = self.lig_var.get().strip()
        out = self.out_var.get().strip()
        strategy = self.strategy_var.get()

        if not rec or not Path(rec).is_dir():
            messagebox.showerror("错误", "请选择有效的大分子文件夹")
            return
        if not lig or not Path(lig).is_dir():
            messagebox.showerror("错误", "请选择有效的小分子父文件夹")
            return
        if not out:
            messagebox.showerror("错误", "请选择输出目录")
            return
        if strategy == "manual" and not Path(
                self.manual_box_var.get()).is_file():
            messagebox.showerror("错误", "手动模式必须选择有效的 .box 文件")
            return

        cfg = {
            "exh": int(self.exh_var.get()),
            "modes": int(self.modes_var.get()),
            "lambda": float(self.lambda_var.get()),
            "max_pockets": int(self.maxp_var.get()),
            "box_size": float(self.boxsz_var.get()),
            "manual_box": self.manual_box_var.get()
                          if strategy == "manual" else None,
        }

        self.log_txt.delete("1.0", "end")
        self._log("=" * 60)
        self._log("开始对接")
        self._log("=" * 60)
        self._log(f"大分子: {rec}")
        self._log(f"小分子: {lig}")
        self._log(f"输出:   {out}")
        self._log(f"策略:   {strategy}")
        self._log(f"参数:   {cfg}")
        self._log(f"项目根: {_ROOT}")
        self._log("\n环境自检:")
        if not check_environment(self._log):
            if not messagebox.askyesno(
                    "环境警告",
                    "部分工具缺失，可能导致运行失败。继续？"):
                return

        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._run_worker,
            args=(rec, lig, out, strategy, cfg), daemon=True)
        self._worker.start()

    def _run_worker(self, rec, lig, out, strategy, cfg):
        try:
            run_all_thread(rec, lig, out, strategy, cfg,
                           self._log_safe, self._set_progress_safe,
                           self._stop_event)
        except Exception as e:
            self._log_safe(f"❌ 运行异常: {e}")
            self._log_safe(traceback.format_exc())
        finally:
            self.root.after(0, self._on_finish)

    def _on_finish(self):
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("就绪")

    def _stop(self):
        self._stop_event.set()
        self._log("⏹ 已发送停止信号，等待当前 case 完成…")


# =====================================================================
# CLI 模式
# =====================================================================
def cli_main():
    import argparse
    p = argparse.ArgumentParser(description="大分子-小分子对接 CLI")
    p.add_argument("--receptor_dir", required=True)
    p.add_argument("--ligand_parent", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--strategy", default="fpocket_rf",
                   choices=[k for k, _ in BOX_STRATEGIES])
    p.add_argument("--exh", type=int, default=32)
    p.add_argument("--modes", type=int, default=20)
    p.add_argument("--lambda_", type=float, default=0.75)
    p.add_argument("--max_pockets", type=int, default=20)
    p.add_argument("--box_size", type=float, default=25.0)
    p.add_argument("--manual_box", default=None)
    a = p.parse_args()

    cfg = {
        "exh": a.exh, "modes": a.modes, "lambda": a.lambda_,
        "max_pockets": a.max_pockets, "box_size": a.box_size,
        "manual_box": a.manual_box,
    }
    check_environment(print)
    run_all_thread(a.receptor_dir, a.ligand_parent, a.out_dir,
                   a.strategy, cfg,
                   lambda m: print(m, flush=True),
                   lambda d, t, c: None, threading.Event())


# =====================================================================
# 入口
# =====================================================================
def main():
    if not resolve_vina():
        print("❌ 未找到 vina", file=sys.stderr)
        print(f"   请放到 {_ROOT / 'tools' / 'vina' / 'vina.exe'}")
        print("   或设置 VINA_BIN 环境变量")
        sys.exit(1)

    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        cli_main()
        return

    root = tk.Tk()
    DockingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()