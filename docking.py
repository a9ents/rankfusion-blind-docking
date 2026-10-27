# pyright: reportAttributeAccessIssue=false
#!/usr/bin/env python3
"""
docking_software.py — 大分子-小分子盲对接软件 GUI
包含：全工具路径手动配置 + 自动保存 (docking_config.json)
"""
import os
import sys
import json
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

import auto_pipeline as ap                    
import run_p2rank_rankfusion as prf           
import run_p2rank_baseline as prb             

# =====================================================================
# 配置文件管理
# =====================================================================
CONFIG_FILE = _ROOT / "docking_config.json"

def load_config():
    if CONFIG_FILE.is_file():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(data):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"保存配置失败: {e}")

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

def resolve_vina():
    v = os.environ.get("VINA_BIN")
    if v and Path(v).is_file(): return v
    p = shutil.which("vina")
    if p: return p
    cand = _ROOT / "tools" / "vina" / "vina.exe"
    if cand.is_file(): return str(cand)
    old_cand = Path(r"E:\tools\vina\vina.exe")
    if old_cand.is_file(): return str(old_cand)
    return ""

def resolve_obabel():
    p = shutil.which("obabel")
    if p: return p
    cand = _ROOT / "tools" / "obabel" / "obabel.exe"
    if cand.is_file(): return str(cand)
    return ""

def resolve_meeko():
    p = shutil.which("mk_prepare_ligand")
    if p: return p
    return ""

def resolve_p2rank_dir():
    v = os.environ.get("P2RANK_DIR")
    if v and (Path(v) / "bin" / "p2rank.jar").is_file(): return Path(v)
    cand = _ROOT / "tools" / "p2rank_2.6-alpha"
    if (cand / "bin" / "p2rank.jar").is_file(): return str(cand)
    return ""

# =====================================================================
# 核心处理逻辑
# =====================================================================
def process_one_case(case, receptor_pdb, ligand_dir, out_root, strategy, cfg, log):
    case_out = Path(out_root) / case
    case_out.mkdir(parents=True, exist_ok=True)

    log(f"\n[{case}] 大分子: {receptor_pdb.name}")
    log(f"[{case}] 小分子: {ligand_dir}")

    rec_pdbqt = case_out / "receptor.pdbqt"
    if not rec_pdbqt.is_file() or rec_pdbqt.stat().st_size < 100:
        log(f"[{case}]   → 转换受体 pdbqt")
        if not ap.pdb_to_pdbqt_receptor(str(receptor_pdb), str(rec_pdbqt)):
            log(f"[{case}]   ✗ 受体转换失败")
            return False

    ligs = sorted([p for p in Path(ligand_dir).iterdir() if p.suffix.lower() in (".pdb", ".sdf", ".mol", ".mol2", ".pdbqt") and p.is_file()])
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

    box_file = case_out / "receptor.box"
    final_center = None
    final_size = None

    if strategy == "blind":
        log(f"[{case}]   → 盲对接 bbox")
        center, size = ap.compute_bounding_box(str(rec_pdbqt), ap.CFG["blind_padding"], ap.CFG["blind_max_size"])
        if center is None:
            log(f"[{case}]   ✗ bbox 计算失败")
            return False
        final_center = center
        final_size = size
        with open(box_file, "w") as f:
            f.write(f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f} {size[0]:.3f} {size[1]:.3f} {size[2]:.3f}\n")

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
            f.write(f"{final_center[0]:.3f} {final_center[1]:.3f} {final_center[2]:.3f} {final_size[0]:.3f} {final_size[1]:.3f} {final_size[2]:.3f}\n")
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
            ok, _ = ap.run_pocket_guided(str(receptor_pdb), str(ligand_dir), str(case_out))
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
        ok, center, size = _run_p2rank_guided(receptor_pdb, ligand_dir, case_out, cfg, log)
        if not ok: return False
        final_center = center
        final_size = size
    else:
        log(f"[{case}]   ✗ 未知策略: {strategy}")
        return False

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
        if ap.run_vina(str(rec_pdbqt), str(lig_pqt), str(out_pqt), final_center, final_size, cfg["exh"], cfg["modes"], 42):
            ok_n += 1
    log(f"[{case}]   ✓ 完成 {ok_n}/{len(lig_pdbqts)}")
    return ok_n > 0

def _run_p2rank_guided(receptor_pdb, ligand_dir, case_out, cfg, log):
    receptor_pdb = Path(receptor_pdb)
    ligand_dir = Path(ligand_dir)
    case_out = Path(case_out)
    case = receptor_pdb.stem

    if not prb.P2RANK_DIR or not (Path(prb.P2RANK_DIR) / "bin" / "p2rank.jar").is_file():
        log(f"[{case}]   ✗ P2Rank 路径无效，请在上方配置 P2Rank 文件夹")
        return False, None, None

    pdb_ligs = list(ligand_dir.glob("*.pdb"))
    if not pdb_ligs:
        sdf_ligs = list(ligand_dir.glob("*.sdf")) + list(ligand_dir.glob("*.mol"))
        if sdf_ligs and ap.OBABEL_BIN:
            probe_pdb = ligand_dir / "_probe_from_sdf.pdb"
            import subprocess
            try:
                subprocess.run([ap.OBABEL_BIN, str(sdf_ligs[0]), "-O", str(probe_pdb)], capture_output=True, timeout=60)
            except Exception: pass
            if probe_pdb.is_file(): pdb_ligs = [probe_pdb]
        if not pdb_ligs:
            log(f"[{case}]   ✗ 无法获得 .pdb 配体，P2Rank 引导失败")
            return False, None, None

    p2rank_out = WORK_ROOT / "p2rank_out"
    p2rank_out.mkdir(parents=True, exist_ok=True)
    prf.P2RANK_OUT_ROOT = p2rank_out
    prf.WORK_ROOT = WORK_ROOT / "p2rank_rf"
    prf.WORK_ROOT.mkdir(parents=True, exist_ok=True)

    has_pred = any(p2rank_out.rglob(f"{case}.pdb_predictions.csv"))
    if not has_pred:
        log(f"[{case}]   → 先运行 P2Rank 检测")
        prb.WORK_ROOT = WORK_ROOT / "p2rank_work"
        prb.WORK_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            _, info = prb.run_one((case, str(receptor_pdb), str(p2rank_out)))
        except Exception as e:
            log(f"[{case}]   ✗ P2Rank 运行异常: {e}")
            return False, None, None
        if info is None:
            log(f"[{case}]   ✗ P2Rank 未产出预测")
            return False, None, None

    tmp_exp = case_out / "_tmp_exp"
    tmp_case = tmp_exp / case
    tmp_case.mkdir(parents=True, exist_ok=True)
    tmp_rec = tmp_case / "receptor.pdb"
    if not tmp_rec.is_file(): shutil.copy2(receptor_pdb, tmp_rec)

    if ligand_dir.name == case:
        ligand_parent_arg = ligand_dir.parent
    else:
        tmp_lig_parent = case_out / "_tmp_lig"
        tmp_lig_case = tmp_lig_parent / case
        tmp_lig_case.mkdir(parents=True, exist_ok=True)
        for lig in ligand_dir.iterdir():
            if lig.is_file() and lig.suffix.lower() == ".pdb":
                dst = tmp_lig_case / lig.name
                if not dst.is_file(): shutil.copy2(lig, dst)
        ligand_parent_arg = tmp_lig_parent

    try:
        _, center, info = prf.process_one(case, str(tmp_exp), str(ligand_parent_arg), str(case_out))
    except Exception as e:
        log(f"[{case}]   ✗ P2Rank 引导异常: {e}")
        return False, None, None

    if center is None:
        log(f"[{case}]   ✗ P2Rank 引导失败: {info}")
        return False, None, None

    src_box = case_out / f"p2rank_rf_{case}.box"
    dst_box = case_out / "receptor.box"
    box_size = cfg.get("box_size", 25.0)
    if src_box.is_file(): shutil.copy(src_box, dst_box)
    else:
        with open(dst_box, "w") as f:
            f.write(f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f} {box_size:.3f} {box_size:.3f} {box_size:.3f}\n")
    return True, tuple(center), (box_size, box_size, box_size)

def run_all_thread(receptor_dir, ligand_parent, out_root, strategy, cfg, log, progress_cb, stop_event):
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
            if process_one_case(case, rec, lig_dir, out_root, strategy, cfg, log):
                ok += 1
            else: fail += 1
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
# GUI 界面
# =====================================================================
class DockingApp:
    def __init__(self, root):
        self.root = root
        root.title("大分子-小分子对接软件 (配置自动保存版)")
        root.geometry("1000x880")
        self._stop_event = threading.Event()
        self._worker = None
        self.config_data = load_config()
        self._build_ui()

    def _build_ui(self):
        # ==================== 0. 工具环境 ====================
        env_frame = ttk.LabelFrame(self.root, text="0. 核心工具配置（修改后点击开始运行会自动保存）", padding=10)
        env_frame.pack(fill="x", padx=10, pady=6)

        # Vina
        ttk.Label(env_frame, text="Vina 程序:").grid(row=0, column=0, sticky="w", pady=2)
        self.vina_var = tk.StringVar(value=self.config_data.get("vina", resolve_vina()))
        ttk.Entry(env_frame, textvariable=self.vina_var, width=75).grid(row=0, column=1, padx=6)
        ttk.Button(env_frame, text="选择…", command=lambda: self._pick_file(self.vina_var)).grid(row=0, column=2)

        # Obabel
        ttk.Label(env_frame, text="Obabel 程序:").grid(row=1, column=0, sticky="w", pady=2)
        self.obabel_var = tk.StringVar(value=self.config_data.get("obabel", resolve_obabel()))
        ttk.Entry(env_frame, textvariable=self.obabel_var, width=75).grid(row=1, column=1, padx=6)
        ttk.Button(env_frame, text="选择…", command=lambda: self._pick_file(self.obabel_var)).grid(row=1, column=2)

        # Meeko
        ttk.Label(env_frame, text="Meeko 程序:").grid(row=2, column=0, sticky="w", pady=2)
        self.meeko_var = tk.StringVar(value=self.config_data.get("meeko", resolve_meeko()))
        ttk.Entry(env_frame, textvariable=self.meeko_var, width=75).grid(row=2, column=1, padx=6)
        ttk.Button(env_frame, text="选择…", command=lambda: self._pick_file(self.meeko_var)).grid(row=2, column=2)

        # P2Rank (文件夹)
        ttk.Label(env_frame, text="P2Rank 文件夹:").grid(row=3, column=0, sticky="w", pady=2)
        self.p2rank_var = tk.StringVar(value=self.config_data.get("p2rank_dir", resolve_p2rank_dir()))
        ttk.Entry(env_frame, textvariable=self.p2rank_var, width=75).grid(row=3, column=1, padx=6)
        ttk.Button(env_frame, text="选择…", command=lambda: self._pick_dir(self.p2rank_var)).grid(row=3, column=2)

        # WSL 环境名
        ttk.Label(env_frame, text="WSL fpocket 环境名:").grid(row=4, column=0, sticky="w", pady=2)
        self.wsl_env_var = tk.StringVar(value=self.config_data.get("wsl_env", "fpocket_env"))
        ttk.Entry(env_frame, textvariable=self.wsl_env_var, width=30).grid(row=4, column=1, padx=6, sticky="w")
        ttk.Label(env_frame, text="(仅限 fpocket_rf 策略)").grid(row=4, column=1, padx=(220,0), sticky="w")

        # ==================== 1. 数据路径 ====================
        top = ttk.LabelFrame(self.root, text="1. 数据路径", padding=10)
        top.pack(fill="x", padx=10, pady=6)

        ttk.Label(top, text="大分子文件夹:").grid(row=0, column=0, sticky="w")
        self.rec_var = tk.StringVar(value=self.config_data.get("rec_dir", ""))
        ttk.Entry(top, textvariable=self.rec_var, width=75).grid(row=0, column=1, padx=6)
        ttk.Button(top, text="选择…", command=self._pick_rec).grid(row=0, column=2)

        ttk.Label(top, text="小分子父文件夹:").grid(row=1, column=0, sticky="w", pady=4)
        self.lig_var = tk.StringVar(value=self.config_data.get("lig_dir", ""))
        ttk.Entry(top, textvariable=self.lig_var, width=75).grid(row=1, column=1, padx=6)
        ttk.Button(top, text="选择…", command=self._pick_lig).grid(row=1, column=2)

        ttk.Label(top, text="输出目录:").grid(row=2, column=0, sticky="w", pady=4)
        self.out_var = tk.StringVar(value=self.config_data.get("out_dir", ""))
        ttk.Entry(top, textvariable=self.out_var, width=75).grid(row=2, column=1, padx=6)
        ttk.Button(top, text="选择…", command=self._pick_out).grid(row=2, column=2)

        # ==================== 2. Box 策略 ====================
        mid = ttk.LabelFrame(self.root, text="2. Box 选择策略", padding=10)
        mid.pack(fill="x", padx=10, pady=6)
        self.strategy_var = tk.StringVar(value=self.config_data.get("strategy", "fpocket_rf"))
        for key, label in BOX_STRATEGIES:
            ttk.Radiobutton(mid, text=label, variable=self.strategy_var, value=key, command=self._on_strategy_change).pack(anchor="w")

        manual_row = ttk.Frame(mid)
        manual_row.pack(fill="x", pady=(6, 0))
        ttk.Label(manual_row, text="  手动 box 文件:").pack(side="left")
        self.manual_box_var = tk.StringVar(value=self.config_data.get("manual_box", ""))
        self.manual_entry = ttk.Entry(manual_row, textvariable=self.manual_box_var, width=55)
        self.manual_entry.pack(side="left", padx=6)
        self.manual_btn = ttk.Button(manual_row, text="选择…", command=self._pick_manual_box)
        self.manual_btn.pack(side="left")
        self._manual_widgets = [self.manual_entry, self.manual_btn]
        self._on_strategy_change()

        # ==================== 3. 对接参数 ====================
        cfg = ttk.LabelFrame(self.root, text="3. 对接参数", padding=10)
        cfg.pack(fill="x", padx=10, pady=6)

        ttk.Label(cfg, text="预设:").grid(row=0, column=0, sticky="w")
        self.preset_var = tk.StringVar(value=PRESETS["standard"]["label"])
        preset_box = ttk.Combobox(cfg, textvariable=self.preset_var, state="readonly", values=[v["label"] for v in PRESETS.values()], width=26)
        preset_box.grid(row=0, column=1, sticky="w", padx=6)
        preset_box.bind("<<ComboboxSelected>>", self._on_preset_change)

        ttk.Label(cfg, text="exhaustiveness:").grid(row=0, column=2, padx=(20, 4), sticky="w")
        self.exh_var = tk.IntVar(value=self.config_data.get("exh", 32))
        ttk.Spinbox(cfg, from_=1, to=128, textvariable=self.exh_var, width=8).grid(row=0, column=3)

        ttk.Label(cfg, text="num_modes:").grid(row=0, column=4, padx=(20, 4), sticky="w")
        self.modes_var = tk.IntVar(value=self.config_data.get("modes", 20))
        ttk.Spinbox(cfg, from_=1, to=100, textvariable=self.modes_var, width=8).grid(row=0, column=5)

        ttk.Label(cfg, text="Rank Fusion λ:").grid(row=1, column=0, sticky="w", pady=4)
        self.lambda_var = tk.DoubleVar(value=self.config_data.get("lambda", 0.75))
        ttk.Spinbox(cfg, from_=0.0, to=10.0, increment=0.05, textvariable=self.lambda_var, width=10).grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(cfg, text="最大口袋数:").grid(row=1, column=2, padx=(20, 4), sticky="w")
        self.maxp_var = tk.IntVar(value=self.config_data.get("max_pockets", 20))
        ttk.Spinbox(cfg, from_=1, to=100, textvariable=self.maxp_var, width=8).grid(row=1, column=3)

        ttk.Label(cfg, text="box 边长 (Å):").grid(row=1, column=4, padx=(20, 4), sticky="w")
        self.boxsz_var = tk.DoubleVar(value=self.config_data.get("box_size", 25.0))
        ttk.Spinbox(cfg, from_=10.0, to=60.0, increment=1.0, textvariable=self.boxsz_var, width=8).grid(row=1, column=5)

        # ==================== 控制按钮 ====================
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=10, pady=(4, 6))
        self.run_btn = ttk.Button(ctrl, text="▶ 开始运行", command=self._start)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(ctrl, text="⏹ 停止", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)

        ttk.Label(ctrl, text="进度:").pack(side="left", padx=(20, 4))
        self.pbar = ttk.Progressbar(ctrl, length=420)
        self.pbar.pack(side="left")
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(ctrl, textvariable=self.status_var).pack(side="left", padx=10)

        # ==================== 日志 ====================
        logf = ttk.LabelFrame(self.root, text="运行日志", padding=6)
        logf.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_txt = scrolledtext.ScrolledText(logf, height=15, font=("Consolas", 9), wrap="word")
        self.log_txt.pack(fill="both", expand=True)

    def _pick_file(self, var):
        f = filedialog.askopenfilename(title="选择可执行文件", filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if f: var.set(f)

    def _pick_dir(self, var):
        d = filedialog.askdirectory(title="选择文件夹")
        if d: var.set(d)

    def _pick_rec(self):
        d = filedialog.askdirectory(title="选择大分子文件夹")
        if d:
            self.rec_var.set(d)
            if not self.out_var.get(): self.out_var.set(str(Path(d).parent / "docking_out"))

    def _pick_lig(self):
        d = filedialog.askdirectory(title="选择小分子父文件夹")
        if d: self.lig_var.set(d)

    def _pick_out(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d: self.out_var.set(d)

    def _pick_manual_box(self):
        f = filedialog.askopenfilename(title="选择 box 文件", filetypes=[("Box files", "*.box"), ("All", "*.*")])
        if f: self.manual_box_var.set(f)

    def _on_strategy_change(self):
        state = "normal" if self.strategy_var.get() == "manual" else "disabled"
        for w in self._manual_widgets: w.configure(state=state)

    def _on_preset_change(self, _e=None):
        label = self.preset_var.get()
        for key, p in PRESETS.items():
            if p["label"] == label and key != "custom":
                self.exh_var.set(p["exh"])
                self.modes_var.set(p["modes"])
                return

    def _log(self, msg):
        self.log_txt.insert("end", msg + "\n")
        self.log_txt.see("end")

    def _log_safe(self, msg): self.root.after(0, self._log, msg)

    def _set_progress(self, done, total, case):
        self.pbar["maximum"] = total
        self.pbar["value"] = done
        self.status_var.set(f"{done}/{total}  {case}")

    def _set_progress_safe(self, done, total, case): self.root.after(0, self._set_progress, done, total, case)

    def _start(self):
        # 收集所有界面上的值
        vina_path = self.vina_var.get().strip()
        obabel_path = self.obabel_var.get().strip()
        meeko_path = self.meeko_var.get().strip()
        p2rank_dir = self.p2rank_var.get().strip()
        wsl_env = self.wsl_env_var.get().strip()
        
        rec = self.rec_var.get().strip()
        lig = self.lig_var.get().strip()
        out = self.out_var.get().strip()
        strategy = self.strategy_var.get()

        if not vina_path or not Path(vina_path).is_file():
            messagebox.showerror("错误", "请指定正确的 Vina 程序路径！")
            return
        if not obabel_path or not Path(obabel_path).is_file():
            messagebox.showerror("错误", "请指定正确的 Obabel 程序路径！")
            return
        if not rec or not Path(rec).is_dir():
            messagebox.showerror("错误", "请选择有效的大分子文件夹")
            return
        if not lig or not Path(lig).is_dir():
            messagebox.showerror("错误", "请选择有效的小分子父文件夹")
            return
        if not out:
            messagebox.showerror("错误", "请选择输出目录")
            return
        if strategy == "manual" and not Path(self.manual_box_var.get()).is_file():
            messagebox.showerror("错误", "手动模式必须选择有效的 .box 文件")
            return

        # 注入到 auto_pipeline
        ap.VINA_BIN = vina_path
        ap.OBABEL_BIN = obabel_path
        if meeko_path:
            ap.MEEKO_BIN = meeko_path
        ap.CFG["pg_wsl_env"] = wsl_env
        
        if p2rank_dir and (Path(p2rank_dir) / "bin" / "p2rank.jar").is_file():
            prb.P2RANK_DIR = Path(p2rank_dir)
        else:
            prb.P2RANK_DIR = None

        cfg = {
            "exh": int(self.exh_var.get()), "modes": int(self.modes_var.get()),
            "lambda": float(self.lambda_var.get()), "max_pockets": int(self.maxp_var.get()),
            "box_size": float(self.boxsz_var.get()),
            "manual_box": self.manual_box_var.get() if strategy == "manual" else None,
        }

        # ---------- 自动保存配置 ----------
        cfg_to_save = {
            "vina": vina_path, "obabel": obabel_path, "meeko": meeko_path,
            "p2rank_dir": p2rank_dir, "wsl_env": wsl_env,
            "rec_dir": rec, "lig_dir": lig, "out_dir": out,
            "strategy": strategy, "manual_box": self.manual_box_var.get(),
            "exh": cfg["exh"], "modes": cfg["modes"], "lambda": cfg["lambda"],
            "max_pockets": cfg["max_pockets"], "box_size": cfg["box_size"]
        }
        save_config(cfg_to_save)

        self.log_txt.delete("1.0", "end")
        self._log("=" * 60)
        self._log("开始对接 (配置已自动保存)")
        self._log("=" * 60)
        self._log(f"Vina:   {vina_path}")
        self._log(f"Obabel: {obabel_path}")
        self._log(f"P2Rank: {p2rank_dir if p2rank_dir else '未配置'}")
        self._log(f"大分子: {rec}\n小分子: {lig}\n输出:   {out}\n策略:   {strategy}\n参数:   {cfg}")

        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run_worker, args=(rec, lig, out, strategy, cfg), daemon=True)
        self._worker.start()

    def _run_worker(self, rec, lig, out, strategy, cfg):
        try:
            run_all_thread(rec, lig, out, strategy, cfg, self._log_safe, self._set_progress_safe, self._stop_event)
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

def main():
    root = tk.Tk()
    DockingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()