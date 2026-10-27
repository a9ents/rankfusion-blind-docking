# pyright: reportAttributeAccessIssue=false
#!/usr/bin/env python3
"""
AF3_Master_Platform.py — AF3 结构与下游干预综合软件平台 (终极致命无错版)
包含：类型安全配置读取、中英双语动态切换、环境配置持久化、全流程自动化
"""
import os
import sys
import json
import shutil
import threading
import traceback
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime

# =====================================================================
# 依赖检查
# =====================================================================
MISSING_DEPS: List[str] = []
try:
    import numpy as np
except ImportError:
    MISSING_DEPS.append("numpy")
try:
    import gemmi
except ImportError:
    MISSING_DEPS.append("gemmi")

# =====================================================================
# 多语言字典 (i18n)
# =====================================================================
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "zh": {
        "app_title": "AF3 综合流水线与分子对接综合平台 (Master OOP)",
        "lang_lbl": "语言:",
        "env_frame": "0. 全局环境配置 (修改后点击任务自动保存)",
        "vina_lbl": "Vina 程序:",
        "obabel_lbl": "Obabel 程序:",
        "meeko_lbl": "Meeko 程序:",
        "p2rank_lbl": "P2Rank 目录:",
        "wsl_lbl": "WSL fpocket 环境名:",
        "root_lbl": "| 项目主目录:",
        "btn_pick": "选择…",
        "tab_convert": "1. 数据转换与修复",
        "tab_dock": "2. 分子对接 (Vina)",
        "tab_probe": "3. 探针与空腔分析",
        "tab_openmm": "4. 物理干预 (OpenMM)",
        
        "conv_desc": "将 AF3 的 CIF 转换为 PDB，并使用 DLPacker 重构侧链",
        "btn_cif2pdb": "▶ 批量 CIF 转 PDB",
        "btn_dlp": "▶ 批量 DLPacker 侧链重建",
        
        "dock_data_frame": "数据路径",
        "rec_lbl": "大分子文件夹:",
        "lig_lbl": "小分子父文件夹:",
        "out_lbl": "输出目录:",
        "dock_strategy_frame": "Box 选择策略",
        "strat_fpocket": "fpocket + Rank Fusion（推荐）",
        "strat_p2rank": "P2Rank + Rank Fusion",
        "strat_manual": "手动指定 box",
        "strat_blind": "盲对接（整蛋白 bbox）",
        "manual_box_lbl": "手动 box 文件:",
        "dock_param_frame": "对接参数",
        "preset_lbl": "预设:",
        "preset_fast": "快速（exh=8, modes=5）",
        "preset_standard": "标准（exh=32, modes=20）",
        "preset_deep": "深度（exh=64, modes=40）",
        "preset_custom": "自定义",
        "exh_lbl": "exhaustiveness:",
        "modes_lbl": "num_modes:",
        "lambda_lbl": "Rank Fusion λ:",
        "maxp_lbl": "最大口袋数:",
        "boxsz_lbl": "box 边长 (Å):",
        "btn_run_dock": "▶ 开始运行对接",
        
        "probe_desc": "计算真值配体位置处的空腔探针半径，并对比不同结构",
        "btn_probe_compare": "▶ 全量探针对比 (AF3 vs OpenMM vs DLP)",
        "btn_probe_survive": "▶ 分析奇迹生还案例",
        
        "openmm_desc": "利用 OpenMM 进行侧链修复、冷热循环退火或配体引导插入",
        "btn_cold": "▶ 冷冻-回暖循环 (cold.py)",
        "btn_fit": "▶ 渐进式物理牵引 (ligand_fit.py)",
        "btn_verify": "▶ 四层热力学验证 (every case.py)",
        
        "btn_stop": "⏹ 停止当前任务",
        "status_ready": "就绪",
        "log_frame": "运行日志",
        
        "msg_missing_deps": "⚠ 警告: 你的环境缺少依赖 {}。涉及底层计算的脚本可能无法运行。\n  请在终端执行: pip install numpy gemmi",
        "msg_err_gemmi": "缺少 gemmi 库，无法执行此任务。\n请在终端运行: pip install gemmi",
        "msg_err_pipeline": "缺少必要的 Python 依赖 (numpy 等)，无法进行对接。\n请在终端运行: pip install numpy pandas gemmi scikit-learn",
        "msg_err_vina": "请指定正确的 Vina 程序路径！",
        "msg_err_path": "请选择有效的大分子或配体文件夹",
        "msg_err_out": "请选择输出目录",
        "msg_err_manual": "手动模式必须选择有效的 .box 文件",
        "msg_save_err": "⚠ 配置保存失败: {}",
        "log_task_start": "▶ 开始新任务 {}",
        "log_task_end": "✅ 任务结束 {}",
        "log_stop": "⏹ 已发送停止信号，等待当前 case 完成…",
        "log_found_cif": "找到 {} 个 CIF 文件，开始转换...",
        "log_conv_success": "  ✓ {}",
        "log_conv_fail": "  ✗ {} 失败: {}",
        "log_no_dlp": "❌ 未安装 dlpacker，无法执行。",
        "log_found_pdb": "找到 {} 个 apo 结构，开始 DLPacker 重建...",
        "log_no_pdb": "❌ {} 下无 *.pdb",
        "log_found_rec": "发现 {} 个大分子",
        "log_no_lig_dir": "[{}/{}] {}   ⚠ 无配体目录，跳过",
        "log_case_start": "\n[{}] 大分子: {}",
        "log_rec_conv": "[{}]   → 转换受体 pdbqt",
        "log_rec_fail": "[{}]   ✗ 受体转换失败",
        "log_lig_empty": "[{}]   ✗ 配体目录无有效文件",
        "log_lig_fail": "[{}]   ✗ 无配体成功转 pdbqt",
        "log_blind_box": "[{}]   → 盲对接 bbox",
        "log_bbox_fail": "[{}]   ✗ bbox 计算失败",
        "log_manual_box": "[{}]   → 使用手动 box",
        "log_fpocket": "[{}]   → fpocket + Rank Fusion",
        "log_fpocket_err": "[{}]   ✗ fpocket 异常: {}",
        "log_fpocket_nobox": "[{}]   ✗ fpocket 未生成 box",
        "log_p2rank": "[{}]   → P2Rank + Rank Fusion",
        "log_p2rank_invalid": "[{}]   ✗ P2Rank 路径无效",
        "log_p2rank_detect": "[{}]   → 先运行 P2Rank 检测",
        "log_p2rank_err": "[{}]   ✗ P2Rank 运行异常: {}",
        "log_p2rank_guide_err": "[{}]   ✗ P2Rank 引导异常: {}",
        "log_p2rank_fail": "[{}]   ✗ P2Rank 引导失败: {}",
        "log_fine_dock": "[{}]   → 精修 exh={} modes={}",
        "log_lig_dock": "[{}]     ({}/{}) {}",
        "log_case_done": "[{}]   ✓ 完成 {}/{}",
        "log_total_done": "\n✅ 完成: 成功 {} / 失败 {} / 总 {}",
        "log_no_script": "❌ 未找到脚本: {}",
        "log_proc_killed": "⏹ 进程已终止",
        "log_exec_err": "执行异常: {}",
        "log_dock_exc": "[{}/{}] {}   ❌ 异常: {}",
        "log_user_abort": "\n⏹ 用户中断",
    },
    "en": {
        "app_title": "AF3 Comprehensive Pipeline & Docking Platform (Master OOP)",
        "lang_lbl": "Language:",
        "env_frame": "0. Global Environment Config (Auto-saved on execution)",
        "vina_lbl": "Vina Executable:",
        "obabel_lbl": "Obabel Executable:",
        "meeko_lbl": "Meeko Executable:",
        "p2rank_lbl": "P2Rank Dir:",
        "wsl_lbl": "WSL fpocket Env:",
        "root_lbl": "| Project Root:",
        "btn_pick": "Browse…",
        "tab_convert": "1. Conversion & Repair",
        "tab_dock": "2. Molecular Docking (Vina)",
        "tab_probe": "3. Probe & Cavity Analysis",
        "tab_openmm": "4. Interventions (OpenMM)",
        
        "conv_desc": "Convert AF3 CIF to PDB and reconstruct sidechains using DLPacker",
        "btn_cif2pdb": "▶ Batch CIF to PDB",
        "btn_dlp": "▶ Batch DLPacker Reconstruction",
        
        "dock_data_frame": "Data Paths",
        "rec_lbl": "Receptor Folder:",
        "lig_lbl": "Ligand Parent Folder:",
        "out_lbl": "Output Dir:",
        "dock_strategy_frame": "Box Strategy",
        "strat_fpocket": "fpocket + Rank Fusion (Recommended)",
        "strat_p2rank": "P2Rank + Rank Fusion",
        "strat_manual": "Manual Box",
        "strat_blind": "Blind Docking (Whole Protein)",
        "manual_box_lbl": "Manual Box File:",
        "dock_param_frame": "Docking Parameters",
        "preset_lbl": "Preset:",
        "preset_fast": "Fast (exh=8, modes=5)",
        "preset_standard": "Standard (exh=32, modes=20)",
        "preset_deep": "Deep (exh=64, modes=40)",
        "preset_custom": "Custom",
        "exh_lbl": "exhaustiveness:",
        "modes_lbl": "num_modes:",
        "lambda_lbl": "Rank Fusion λ:",
        "maxp_lbl": "Max Pockets:",
        "boxsz_lbl": "Box Size (Å):",
        "btn_run_dock": "▶ Start Docking",
        
        "probe_desc": "Calculate cavity probe radius at true ligand center and compare structures",
        "btn_probe_compare": "▶ Compare All Probes (AF3 vs OpenMM vs DLP)",
        "btn_probe_survive": "▶ Analyze Survivor Cases",
        
        "openmm_desc": "OpenMM sidechain repair, cryo-warming annealing, or guided ligand insertion",
        "btn_cold": "▶ Cryo-Warming (cold.py)",
        "btn_fit": "▶ Progressive Physical Pull (ligand_fit.py)",
        "btn_verify": "▶ 4-Layer Thermodynamic Verify (every case.py)",
        
        "btn_stop": "⏹ Stop Current Task",
        "status_ready": "Ready",
        "log_frame": "Execution Log",
        
        "msg_missing_deps": "⚠ Warning: Missing dependencies {}. Computational scripts may fail.\n  Run: pip install numpy gemmi",
        "msg_err_gemmi": "Missing gemmi library.\nRun: pip install gemmi",
        "msg_err_pipeline": "Missing core Python dependencies.\nRun: pip install numpy pandas gemmi scikit-learn",
        "msg_err_vina": "Please specify a valid Vina path!",
        "msg_err_path": "Please select valid receptor or ligand folders",
        "msg_err_out": "Please select an output directory",
        "msg_err_manual": "Manual mode requires a valid .box file",
        "msg_save_err": "⚠ Config save failed: {}",
        "log_task_start": "▶ Start new task {}",
        "log_task_end": "✅ Task finished {}",
        "log_stop": "⏹ Stop signal sent, waiting for current case…",
        "log_found_cif": "Found {} CIF files, converting...",
        "log_conv_success": "  ✓ {}",
        "log_conv_fail": "  ✗ {} failed: {}",
        "log_no_dlp": "❌ dlpacker not installed.",
        "log_found_pdb": "Found {} apo structures, reconstructing...",
        "log_no_pdb": "❌ No *.pdb in {}",
        "log_found_rec": "Found {} receptors",
        "log_no_lig_dir": "[{}/{}] {}   ⚠ No ligand dir, skipping",
        "log_case_start": "\n[{}] Receptor: {}",
        "log_rec_conv": "[{}]   → Converting receptor pdbqt",
        "log_rec_fail": "[{}]   ✗ Receptor conversion failed",
        "log_lig_empty": "[{}]   ✗ No valid ligand files",
        "log_lig_fail": "[{}]   ✗ No ligand converted",
        "log_blind_box": "[{}]   → Blind docking bbox",
        "log_bbox_fail": "[{}]   ✗ bbox calculation failed",
        "log_manual_box": "[{}]   → Using manual box",
        "log_fpocket": "[{}]   → fpocket + Rank Fusion",
        "log_fpocket_err": "[{}]   ✗ fpocket exception: {}",
        "log_fpocket_nobox": "[{}]   ✗ fpocket generated no box",
        "log_p2rank": "[{}]   → P2Rank + Rank Fusion",
        "log_p2rank_invalid": "[{}]   ✗ Invalid P2Rank path",
        "log_p2rank_detect": "[{}]   → Running P2Rank detection first",
        "log_p2rank_err": "[{}]   ✗ P2Rank execution failed: {}",
        "log_p2rank_guide_err": "[{}]   ✗ P2Rank guided exception: {}",
        "log_p2rank_fail": "[{}]   ✗ P2Rank guided failed: {}",
        "log_fine_dock": "[{}]   → Fine docking exh={} modes={}",
        "log_lig_dock": "[{}]     ({}/{}) {}",
        "log_case_done": "[{}]   ✓ Done {}/{}",
        "log_total_done": "\n✅ Complete: Success {} / Fail {} / Total {}",
        "log_no_script": "❌ Script not found: {}",
        "log_proc_killed": "⏹ Process terminated",
        "log_exec_err": "Execution error: {}",
        "log_dock_exc": "[{}/{}] {}   ❌ Exception: {}",
        "log_user_abort": "\n⏹ User aborted",
    }
}

_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = _ROOT / "master_platform_config.json"

for _sub in ("pipeline", "src", "analysis", _ROOT / "NEW"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

HAS_PIPELINE = False
try:
    import auto_pipeline as ap
    import run_p2rank_rankfusion as prf
    import run_p2rank_baseline as prb
    HAS_PIPELINE = True
except ImportError as e:
    print(f"Warning: Pipeline import failed ({e})")

WORK_ROOT = _ROOT / "work"

class AF3MasterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("1100x920")
        
        # 状态变量
        self.config_data: Dict[str, Any] = self._load_config()
        self._stop_event = threading.Event()
        self.is_running = False
        self.current_lang: str = str(self.config_data.get("lang", "zh"))
        
        # 构建 UI
        self._build_ui()
        self._on_strategy_change()

    # ================= 语言与配置管理 =================
    def tr(self, key: str, *args: Any) -> str:
        """获取当前语言的翻译文本，强制返回 str"""
        lang_dict = TRANSLATIONS.get(self.current_lang, TRANSLATIONS["zh"])
        text = lang_dict.get(key, key)
        if args:
            return text.format(*args)
        return text

    def _load_config(self) -> Dict[str, Any]:
        if CONFIG_FILE.is_file():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return cast(Dict[str, Any], data)
            except Exception:
                pass
        return {}

    def _get_str(self, key: str, default: str = "") -> str:
        """安全读取字符串配置"""
        val = self.config_data.get(key)
        return str(val) if val is not None else default

    def _get_int(self, key: str, default: int) -> int:
        """安全读取整数配置"""
        val = self.config_data.get(key)
        try:
            return int(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    def _get_float(self, key: str, default: float) -> float:
        """安全读取浮点数配置"""
        val = self.config_data.get(key)
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    def _save_config(self):
        cfg = {
            "lang": self.current_lang,
            "vina": self.vina_var.get(), "obabel": self.obabel_var.get(),
            "meeko": self.meeko_var.get(), "p2rank": self.p2rank_var.get(),
            "wsl_env": self.wsl_env_var.get(), "root_dir": self.root_dir_var.get(),
            "dock_rec": self.dock_rec_var.get(), "dock_lig": self.dock_lig_var.get(),
            "dock_out": self.dock_out_var.get(), "strategy": self.strategy_var.get(),
            "manual_box": self.manual_box_var.get(), "preset": self.preset_var.get(),
            "exh": self.exh_var.get(), "modes": self.modes_var.get(),
            "lambda": self.lambda_var.get(), "max_pockets": self.maxp_var.get(),
            "box_size": self.boxsz_var.get()
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.log(self.tr("msg_save_err", str(e)))

    def _switch_lang(self, event=None):
        lang_map = {"中文": "zh", "English": "en"}
        self.current_lang = lang_map.get(self.lang_combo.get(), "zh")
        self.title(self.tr("app_title"))
        
        self._update_widget_lang(self)
        
        self.notebook.tab(0, text=self.tr("tab_convert"))
        self.notebook.tab(1, text=self.tr("tab_dock"))
        self.notebook.tab(2, text=self.tr("tab_probe"))
        self.notebook.tab(3, text=self.tr("tab_openmm"))
        
        for key, rb in self.strategy_radios.items():
            rb.configure(text=self.tr(f"strat_{key}"))
            
        self._update_preset_combo()
        self._save_config()

    def _update_widget_lang(self, widget):
        if hasattr(widget, "trans_key"):
            key = getattr(widget, "trans_key")
            widget.configure(text=self.tr(key)) # type: ignore
        for child in widget.winfo_children():
            self._update_widget_lang(child)

    def _update_preset_combo(self):
        presets = {
            "fast": self.tr("preset_fast"),
            "standard": self.tr("preset_standard"),
            "deep": self.tr("preset_deep"),
            "custom": self.tr("preset_custom"),
        }
        self.preset_box.configure(values=[presets[k] for k in ("fast", "standard", "deep", "custom")])
        current_key = self.config_data.get("preset", "standard")
        if not current_key in presets: 
            current_key = "standard"
        self.preset_var.set(presets[current_key])

    # ================= UI 构建 =================
    def _build_ui(self):
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", padx=10, pady=5)
        
        lbl_lang = ttk.Label(top_bar, text=self.tr("lang_lbl"))
        lbl_lang.trans_key = "lang_lbl"
        lbl_lang.pack(side="right")
        
        self.lang_combo = ttk.Combobox(top_bar, values=["中文", "English"], width=8, state="readonly")
        self.lang_combo.set("中文" if self.current_lang == "zh" else "English")
        self.lang_combo.pack(side="right", padx=5)
        self.lang_combo.bind("<<ComboboxSelected>>", self._switch_lang)

        env_frame = ttk.LabelFrame(self, text=self.tr("env_frame"), padding=10)
        env_frame.trans_key = "env_frame"
        env_frame.pack(fill="x", padx=10, pady=5)

        self.vina_var = tk.StringVar(value=self._get_str("vina", self._resolve_vina()))
        self.obabel_var = tk.StringVar(value=self._get_str("obabel", self._resolve_obabel()))
        self.meeko_var = tk.StringVar(value=self._get_str("meeko", self._resolve_meeko()))
        self.p2rank_var = tk.StringVar(value=self._get_str("p2rank", self._resolve_p2rank()))
        self.wsl_env_var = tk.StringVar(value=self._get_str("wsl_env", "fpocket_env"))
        self.root_dir_var = tk.StringVar(value=self._get_str("root_dir", r"E:\AF3环境\ALL-DATA"))

        self._add_config_row(env_frame, 0, "vina_lbl", self.vina_var, "file")
        self._add_config_row(env_frame, 1, "obabel_lbl", self.obabel_var, "file")
        self._add_config_row(env_frame, 2, "meeko_lbl", self.meeko_var, "file")
        self._add_config_row(env_frame, 3, "p2rank_lbl", self.p2rank_var, "dir")
        
        row4 = ttk.Frame(env_frame)
        row4.grid(row=4, column=0, columnspan=3, sticky="w", pady=2)
        lbl = ttk.Label(row4, text=self.tr("wsl_lbl"))
        lbl.trans_key = "wsl_lbl"
        lbl.pack(side="left")
        ttk.Entry(row4, textvariable=self.wsl_env_var, width=20).pack(side="left", padx=5)
        
        lbl2 = ttk.Label(row4, text=self.tr("root_lbl"))
        lbl2.trans_key = "root_lbl"
        lbl2.pack(side="left")
        ttk.Entry(row4, textvariable=self.root_dir_var, width=45).pack(side="left", padx=5)
        btn2 = ttk.Button(row4, text=self.tr("btn_pick"), command=lambda: self._pick_path(self.root_dir_var, "dir"))
        btn2.trans_key = "btn_pick"
        btn2.pack(side="left")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        self.tab_convert = ttk.Frame(self.notebook, padding=10)
        self.tab_dock = ttk.Frame(self.notebook, padding=10)
        self.tab_probe = ttk.Frame(self.notebook, padding=10)
        self.tab_openmm = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_convert, text=self.tr("tab_convert"))
        self.notebook.add(self.tab_dock, text=self.tr("tab_dock"))
        self.notebook.add(self.tab_probe, text=self.tr("tab_probe"))
        self.notebook.add(self.tab_openmm, text=self.tr("tab_openmm"))

        self._build_tab_convert()
        self._build_tab_dock()
        self._build_tab_probe()
        self._build_tab_openmm()

        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill="x", padx=10, pady=5)
        self.btn_stop = ttk.Button(ctrl_frame, text=self.tr("btn_stop"), command=self._stop_tasks, state="disabled")
        self.btn_stop.trans_key = "btn_stop"
        self.btn_stop.pack(side="left")
        
        self.pbar = ttk.Progressbar(ctrl_frame, length=400)
        self.pbar.pack(side="left", padx=20)
        
        self.status_var = tk.StringVar(value=self.tr("status_ready"))
        lbl_status = ttk.Label(ctrl_frame, textvariable=self.status_var)
        lbl_status.pack(side="left")

        log_frame = ttk.LabelFrame(self, text=self.tr("log_frame"), padding=5)
        log_frame.trans_key = "log_frame"
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_txt = scrolledtext.ScrolledText(log_frame, height=15, font=("Consolas", 9), wrap="word", bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
        self.log_txt.pack(fill="both", expand=True)
        
        if MISSING_DEPS:
            self.log(self.tr("msg_missing_deps", ", ".join(MISSING_DEPS)))

    def _add_config_row(self, parent, row, trans_key, var, pick_type):
        lbl = ttk.Label(parent, text=self.tr(trans_key))
        lbl.trans_key = trans_key
        lbl.grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=80).grid(row=row, column=1, padx=6)
        btn = ttk.Button(parent, text=self.tr("btn_pick"), command=lambda: self._pick_path(var, pick_type))
        btn.trans_key = "btn_pick"
        btn.grid(row=row, column=2)

    def _build_tab_convert(self):
        lbl = ttk.Label(self.tab_convert, text=self.tr("conv_desc"))
        lbl.trans_key = "conv_desc"
        lbl.pack(anchor="w", pady=10)
        
        btn1 = ttk.Button(self.tab_convert, text=self.tr("btn_cif2pdb"), command=self._task_cif_to_pdb)
        btn1.trans_key = "btn_cif2pdb"
        btn1.pack(anchor="w", pady=5)
        
        btn2 = ttk.Button(self.tab_convert, text=self.tr("btn_dlp"), command=self._task_dlp_reconstruct)
        btn2.trans_key = "btn_dlp"
        btn2.pack(anchor="w", pady=5)

    def _build_tab_dock(self):
        top = ttk.LabelFrame(self.tab_dock, text=self.tr("dock_data_frame"), padding=5)
        top.trans_key = "dock_data_frame"
        top.pack(fill="x", pady=5)

        self.dock_rec_var = tk.StringVar(value=self._get_str("dock_rec"))
        self.dock_lig_var = tk.StringVar(value=self._get_str("dock_lig"))
        self.dock_out_var = tk.StringVar(value=self._get_str("dock_out"))

        for i, (key, var) in enumerate([("rec_lbl", self.dock_rec_var), ("lig_lbl", self.dock_lig_var), ("out_lbl", self.dock_out_var)]):
            lbl = ttk.Label(top, text=self.tr(key))
            lbl.trans_key = key
            lbl.grid(row=i, column=0, sticky="w", pady=4)
            ttk.Entry(top, textvariable=var, width=65).grid(row=i, column=1, padx=6)
            btn = ttk.Button(top, text=self.tr("btn_pick"), command=lambda v=var: self._pick_path(v, "dir"))
            btn.trans_key = "btn_pick"
            btn.grid(row=i, column=2)

        mid = ttk.LabelFrame(self.tab_dock, text=self.tr("dock_strategy_frame"), padding=5)
        mid.trans_key = "dock_strategy_frame"
        mid.pack(fill="x", pady=5)
        
        self.strategy_var = tk.StringVar(value=self._get_str("strategy", "fpocket_rf"))
        self.strategy_radios = {}
        strategies = ["fpocket_rf", "p2rank_rf", "manual", "blind"]
        for key in strategies:
            rb = ttk.Radiobutton(mid, text=self.tr(f"strat_{key}"), variable=self.strategy_var, value=key, command=self._on_strategy_change)
            rb.pack(anchor="w")
            self.strategy_radios[key] = rb

        manual_row = ttk.Frame(mid)
        manual_row.pack(fill="x", pady=(6, 0))
        lbl = ttk.Label(manual_row, text=self.tr("manual_box_lbl"))
        lbl.trans_key = "manual_box_lbl"
        lbl.pack(side="left")
        self.manual_box_var = tk.StringVar(value=self._get_str("manual_box"))
        self.manual_entry = ttk.Entry(manual_row, textvariable=self.manual_box_var, width=55)
        self.manual_entry.pack(side="left", padx=6)
        self.manual_btn = ttk.Button(manual_row, text=self.tr("btn_pick"), command=lambda: self._pick_path(self.manual_box_var, "file"))
        self.manual_btn.trans_key = "btn_pick"
        self.manual_btn.pack(side="left")
        self._manual_widgets = [self.manual_entry, self.manual_btn]

        cfg = ttk.LabelFrame(self.tab_dock, text=self.tr("dock_param_frame"), padding=5)
        cfg.trans_key = "dock_param_frame"
        cfg.pack(fill="x", pady=5)

        lbl = ttk.Label(cfg, text=self.tr("preset_lbl"))
        lbl.trans_key = "preset_lbl"
        lbl.grid(row=0, column=0, sticky="w")
        
        self.preset_var = tk.StringVar(value=self._get_str("preset", "standard"))
        self.preset_box = ttk.Combobox(cfg, textvariable=self.preset_var, state="readonly", width=26)
        self.preset_box.grid(row=0, column=1, sticky="w", padx=6)
        self.preset_box.bind("<<ComboboxSelected>>", self._on_preset_change)
        
        # 使用安全读取
        self.exh_var = tk.IntVar(value=self._get_int("exh", 32))
        self.modes_var = tk.IntVar(value=self._get_int("modes", 20))
        self.lambda_var = tk.DoubleVar(value=self._get_float("lambda", 0.75))
        self.maxp_var = tk.IntVar(value=self._get_int("max_pockets", 20))
        self.boxsz_var = tk.DoubleVar(value=self._get_float("box_size", 25.0))

        ttk.Label(cfg, text=self.tr("exh_lbl")).grid(row=0, column=2, padx=(20, 4), sticky="w")
        ttk.Spinbox(cfg, from_=1, to=128, textvariable=self.exh_var, width=8).grid(row=0, column=3)

        ttk.Label(cfg, text=self.tr("modes_lbl")).grid(row=0, column=4, padx=(20, 4), sticky="w")
        ttk.Spinbox(cfg, from_=1, to=100, textvariable=self.modes_var, width=8).grid(row=0, column=5)

        ttk.Label(cfg, text=self.tr("lambda_lbl")).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Spinbox(cfg, from_=0.0, to=10.0, increment=0.05, textvariable=self.lambda_var, width=10).grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(cfg, text=self.tr("maxp_lbl")).grid(row=1, column=2, padx=(20, 4), sticky="w")
        ttk.Spinbox(cfg, from_=1, to=100, textvariable=self.maxp_var, width=8).grid(row=1, column=3)

        ttk.Label(cfg, text=self.tr("boxsz_lbl")).grid(row=1, column=4, padx=(20, 4), sticky="w")
        ttk.Spinbox(cfg, from_=10.0, to=60.0, increment=1.0, textvariable=self.boxsz_var, width=8).grid(row=1, column=5)

        btn_run = ttk.Button(self.tab_dock, text=self.tr("btn_run_dock"), command=self._start_docking)
        btn_run.trans_key = "btn_run_dock"
        btn_run.pack(anchor="w", pady=15)

    def _build_tab_probe(self):
        lbl = ttk.Label(self.tab_probe, text=self.tr("probe_desc"))
        lbl.trans_key = "probe_desc"
        lbl.pack(anchor="w", pady=10)
        
        btn1 = ttk.Button(self.tab_probe, text=self.tr("btn_probe_compare"), command=lambda: self._task_run_external_script("compare_all_probes(2).py"))
        btn1.trans_key = "btn_probe_compare"
        btn1.pack(anchor="w", pady=5)
        
        btn2 = ttk.Button(self.tab_probe, text=self.tr("btn_probe_survive"), command=lambda: self._task_run_external_script("analyze_survivors(1).py"))
        btn2.trans_key = "btn_probe_survive"
        btn2.pack(anchor="w", pady=5)

    def _build_tab_openmm(self):
        lbl = ttk.Label(self.tab_openmm, text=self.tr("openmm_desc"))
        lbl.trans_key = "openmm_desc"
        lbl.pack(anchor="w", pady=10)
        
        btn1 = ttk.Button(self.tab_openmm, text=self.tr("btn_cold"), command=lambda: self._task_run_external_script("cold.py"))
        btn1.trans_key = "btn_cold"
        btn1.pack(anchor="w", pady=5)
        
        btn2 = ttk.Button(self.tab_openmm, text=self.tr("btn_fit"), command=lambda: self._task_run_external_script("ligand_fit.py"))
        btn2.trans_key = "btn_fit"
        btn2.pack(anchor="w", pady=5)
        
        btn3 = ttk.Button(self.tab_openmm, text=self.tr("btn_verify"), command=lambda: self._task_run_external_script("every case.py"))
        btn3.trans_key = "btn_verify"
        btn3.pack(anchor="w", pady=5)

    # ================= 通用交互 =================
    def _resolve_vina(self) -> str:
        v = os.environ.get("VINA_BIN")
        if v and Path(v).is_file(): return v
        p = shutil.which("vina")
        if p: return p
        return ""

    def _resolve_obabel(self) -> str:
        p = shutil.which("obabel")
        if p: return p
        return ""

    def _resolve_meeko(self) -> str:
        p = shutil.which("mk_prepare_ligand")
        if p: return p
        return ""

    def _resolve_p2rank(self) -> str:
        v = os.environ.get("P2RANK_DIR")
        if v and (Path(v) / "bin" / "p2rank.jar").is_file(): return v
        cand = _ROOT / "tools" / "p2rank_2.6-alpha"
        if (cand / "bin" / "p2rank.jar").is_file(): return str(cand)
        return ""

    def _pick_path(self, var: tk.StringVar, pick_type: str):
        if pick_type == "dir":
            p = filedialog.askdirectory()
        else:
            p = filedialog.askopenfilename()
        if p: var.set(p)

    def _on_strategy_change(self):
        state = "normal" if self.strategy_var.get() == "manual" else "disabled"
        for w in getattr(self, '_manual_widgets', []):
            w.configure(state=state)

    def _on_preset_change(self, _e=None):
        label = self.preset_var.get()
        presets = {
            "fast": {"exh": 8, "modes": 5},
            "standard": {"exh": 32, "modes": 20},
            "deep": {"exh": 64, "modes": 40}
        }
        for k, v in presets.items():
            if self.tr(f"preset_{k}") == label:
                self.exh_var.set(v["exh"])
                self.modes_var.set(v["modes"])
                self.config_data["preset"] = k
                return

    def log(self, msg: str):
        self.after(0, lambda: (self.log_txt.insert("end", msg + "\n"), self.log_txt.see("end")))

    def set_progress(self, done: int, total: int, case: str):
        def _update():
            self.pbar["maximum"] = total
            self.pbar["value"] = done
            self.status_var.set(f"{done}/{total} {case}")
        self.after(0, _update)

    def _start_task(self):
        self._save_config()
        self.is_running = True
        self._stop_event.clear()
        self.after(0, lambda: self.btn_stop.configure(state="normal"))
        self.log("\n" + "="*60 + "\n" + self.tr("log_task_start", datetime.now().strftime('%H:%M:%S')))

    def _finish_task(self):
        self.is_running = False
        self.after(0, lambda: self.btn_stop.configure(state="disabled"))
        self.after(0, lambda: self.status_var.set(self.tr("status_ready")))
        self.log(self.tr("log_task_end", datetime.now().strftime('%H:%M:%S')) + "\n" + "="*60)

    def _stop_tasks(self):
        self._stop_event.set()
        self.log(self.tr("log_stop"))

    # ================= 业务逻辑 =================
    def _task_cif_to_pdb(self):
        if self.is_running: return
        if not gemmi:
            messagebox.showerror("Error", self.tr("msg_err_gemmi"))
            return
        def run():
            self._start_task()
            root = Path(self.root_dir_var.get())
            src = root / "DATA" / "AF3_workflow"
            dst = root / "DATA" / "AF3-PBD"
            dst.mkdir(parents=True, exist_ok=True)
            cifs = list(src.glob("*/*_MODEL.CIF")) + list(src.glob("*/*.cif"))
            self.log(self.tr("log_found_cif", len(cifs)))
            for cif in cifs:
                if self._stop_event.is_set(): break
                case = cif.parent.name
                out_pdb = dst / f"{case}_apo.pdb"
                try:
                    st = gemmi.read_structure(str(cif))
                    for model in st:
                        for chain in model:
                            to_del = [i for i, r in enumerate(chain) if r.entity_type != gemmi.EntityType.Polymer]
                            for i in reversed(to_del): del chain[i]
                    st.remove_empty_chains()
                    st.write_pdb(str(out_pdb))
                    self.log(self.tr("log_conv_success", case))
                except Exception as e:
                    self.log(self.tr("log_conv_fail", case, str(e)))
            self._finish_task()
        threading.Thread(target=run, daemon=True).start()

    def _task_dlp_reconstruct(self):
        if self.is_running: return
        def run():
            self._start_task()
            try:
                from dlpacker import DLPacker
            except ImportError:
                self.log(self.tr("log_no_dlp"))
                self._finish_task()
                return
            root = Path(self.root_dir_var.get())
            src = root / "DATA" / "AF3-PBD"
            dst = root / "DATA" / "AF3-PBD-DLP"
            dst.mkdir(parents=True, exist_ok=True)
            pdbs = list(src.glob("*_apo.pdb"))
            self.log(self.tr("log_found_pdb", len(pdbs)))
            for pdb in pdbs:
                if self._stop_event.is_set(): break
                case = pdb.stem.replace("_apo", "")
                out_pdb = dst / f"{case}_dlp.pdb"
                if out_pdb.exists(): continue
                try:
                    dlp = DLPacker(str(pdb))
                    dlp.reconstruct_protein(order='sequence', output_filename=str(out_pdb))
                    self.log(self.tr("log_conv_success", case))
                except Exception as e:
                    self.log(self.tr("log_conv_fail", case, str(e)))
            self._finish_task()
        threading.Thread(target=run, daemon=True).start()

    def _start_docking(self):
        if self.is_running: return
        if not HAS_PIPELINE:
            messagebox.showerror("Error", self.tr("msg_err_pipeline"))
            return
        vina_path, obabel_path = self.vina_var.get().strip(), self.obabel_var.get().strip()
        rec, lig, out = self.dock_rec_var.get().strip(), self.dock_lig_var.get().strip(), self.dock_out_var.get().strip()
        strategy = self.strategy_var.get()

        if not vina_path or not Path(vina_path).is_file():
            messagebox.showerror("Error", self.tr("msg_err_vina")); return
        if not rec or not Path(rec).is_dir() or not lig or not Path(lig).is_dir():
            messagebox.showerror("Error", self.tr("msg_err_path")); return
        if not out:
            messagebox.showerror("Error", self.tr("msg_err_out")); return
        if strategy == "manual" and not Path(self.manual_box_var.get()).is_file():
            messagebox.showerror("Error", self.tr("msg_err_manual")); return

        ap.VINA_BIN = vina_path
        ap.OBABEL_BIN = obabel_path
        if self.meeko_var.get(): ap.MEEKO_BIN = self.meeko_var.get()
        ap.CFG["pg_wsl_env"] = self.wsl_env_var.get()
        
        p2rank_dir = self.p2rank_var.get().strip()
        if p2rank_dir and (Path(p2rank_dir) / "bin" / "p2rank.jar").is_file():
            prb.P2RANK_DIR = Path(p2rank_dir)
        else:
            prb.P2RANK_DIR = None

        cfg = {
            "exh": self._get_int("exh", 32), 
            "modes": self._get_int("modes", 20),
            "lambda": self._get_float("lambda", 0.75), 
            "max_pockets": self._get_int("max_pockets", 20),
            "box_size": self._get_float("box_size", 25.0),
            "manual_box": self.manual_box_var.get() if strategy == "manual" else None,
        }

        def run():
            self._start_task()
            pdbs = sorted(p for p in Path(rec).glob("*.pdb") if p.is_file())
            if not pdbs:
                self.log(self.tr("log_no_pdb", rec))
                self._finish_task(); return
            self.log(self.tr("log_found_rec", len(pdbs)))
            ok = fail = 0
            for i, p in enumerate(pdbs, 1):
                if self._stop_event.is_set():
                    self.log(self.tr("log_user_abort")); break
                case = p.stem
                lig_dir = Path(lig) / case
                if not lig_dir.is_dir():
                    self.log(self.tr("log_no_lig_dir", i, len(pdbs), case)); fail += 1
                    self.set_progress(i, len(pdbs), case); continue
                self.set_progress(i - 1, len(pdbs), case)
                try:
                    if self._process_one_case(case, p, lig_dir, Path(out), strategy, cfg): ok += 1
                    else: fail += 1
                except Exception as e:
                    self.log(self.tr("log_dock_exc", i, len(pdbs), case, str(e))); fail += 1
                self.set_progress(i, len(pdbs), case)
            self.log(self.tr("log_total_done", ok, fail, len(pdbs)))
            self._finish_task()
        threading.Thread(target=run, daemon=True).start()

    def _process_one_case(self, case: str, receptor_pdb: Path, ligand_dir: Path, out_root: Path, strategy: str, cfg: Dict[str, Any]) -> bool:
        case_out = out_root / case
        case_out.mkdir(parents=True, exist_ok=True)
        self.log(self.tr("log_case_start", case, receptor_pdb.name))
        
        rec_pdbqt = case_out / "receptor.pdbqt"
        if not rec_pdbqt.is_file() or rec_pdbqt.stat().st_size < 100:
            self.log(self.tr("log_rec_conv", case))
            if not ap.pdb_to_pdbqt_receptor(str(receptor_pdb), str(rec_pdbqt)):
                self.log(self.tr("log_rec_fail", case)); return False

        ligs = sorted([p for p in ligand_dir.iterdir() if p.suffix.lower() in (".pdb", ".sdf", ".mol", ".mol2", ".pdbqt") and p.is_file()])
        if not ligs:
            self.log(self.tr("log_lig_empty", case)); return False

        lig_work = case_out / "ligands_pdbqt"
        lig_work.mkdir(exist_ok=True)
        lig_pdbqts = []
        for lig in ligs:
            out_pqt = lig_work / f"{lig.stem}.pdbqt"
            if not out_pqt.is_file() or out_pqt.stat().st_size < 100:
                if not ap.pdb_to_pdbqt_ligand(str(lig), str(out_pqt)): continue
            lig_pdbqts.append(out_pqt)
        if not lig_pdbqts:
            self.log(self.tr("log_lig_fail", case)); return False

        box_file = case_out / "receptor.box"
        final_center = final_size = None

        if strategy == "blind":
            self.log(self.tr("log_blind_box", case))
            center, size = ap.compute_bounding_box(str(rec_pdbqt), ap.CFG["blind_padding"], ap.CFG["blind_max_size"])
            if center is None:
                self.log(self.tr("log_bbox_fail", case)); return False
            final_center, final_size = center, size
            with open(box_file, "w") as f:
                f.write(f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f} {size[0]:.3f} {size[1]:.3f} {size[2]:.3f}\n")

        elif strategy == "manual":
            manual_src = cfg.get("manual_box")
            if not manual_src or not Path(manual_src).is_file():
                self.log(self.tr("msg_err_manual"))
                return False
            with open(str(manual_src)) as f: 
                parts = f.read().split()
            final_center = tuple(map(float, parts[:3]))
            final_size = tuple(map(float, parts[3:6]))
            with open(box_file, "w") as f:
                f.write(f"{final_center[0]:.3f} {final_center[1]:.3f} {final_center[2]:.3f} {final_size[0]:.3f} {final_size[1]:.3f} {final_size[2]:.3f}\n")
            self.log(self.tr("log_manual_box", case))

        elif strategy == "fpocket_rf":
            self.log(self.tr("log_fpocket", case))
            ap.CFG["blind_mode"] = "pocket_guided"
            ap.CFG["pg_rank_lambda"] = cfg.get("lambda", 0.75)
            ap.CFG["pg_quick_exh"] = 4
            ap.CFG["pg_quick_modes"] = 3
            ap.CFG["pg_max_pockets"] = cfg.get("max_pockets", 20)
            ap.CFG["pg_box_size"] = cfg.get("box_size", 25.0)
            try:
                ok, _ = ap.run_pocket_guided(str(receptor_pdb), str(ligand_dir), str(case_out))
            except Exception as e:
                self.log(self.tr("log_fpocket_err", case, str(e))); return False
            if not ok or not box_file.is_file():
                self.log(self.tr("log_fpocket_nobox", case)); return False
            with open(box_file) as f: parts = f.read().split()
            final_center = tuple(map(float, parts[:3]))
            final_size = tuple(map(float, parts[3:6]))

        elif strategy == "p2rank_rf":
            self.log(self.tr("log_p2rank", case))
            ok, center, size = self._run_p2rank_guided(receptor_pdb, ligand_dir, case_out, cfg)
            if not ok: return False
            final_center, final_size = center, size

        self.log(self.tr("log_fine_dock", case, cfg['exh'], cfg['modes']))
        fine_dir = case_out / "docking_results"
        fine_dir.mkdir(exist_ok=True)
        ok_n = 0
        for i, lig_pqt in enumerate(lig_pdbqts, 1):
            if self._stop_event.is_set(): break
            out_pqt = fine_dir / f"docking_{lig_pqt.stem}.pdbqt"
            if out_pqt.is_file() and out_pqt.stat().st_size > 100:
                ok_n += 1; continue
            self.log(self.tr("log_lig_dock", case, i, len(lig_pdbqts), lig_pqt.stem))
            if ap.run_vina(str(rec_pdbqt), str(lig_pqt), str(out_pqt), final_center, final_size, cfg["exh"], cfg["modes"], 42):
                ok_n += 1
        self.log(self.tr("log_case_done", case, ok_n, len(lig_pdbqts)))
        return ok_n > 0

    def _run_p2rank_guided(self, receptor_pdb: Path, ligand_dir: Path, case_out: Path, cfg: Dict[str, Any]) -> tuple:
        case = receptor_pdb.stem
        if not prb.P2RANK_DIR or not (Path(prb.P2RANK_DIR) / "bin" / "p2rank.jar").is_file():
            self.log(self.tr("log_p2rank_invalid", case)); return False, None, None
        p2rank_out = WORK_ROOT / "p2rank_out"
        p2rank_out.mkdir(parents=True, exist_ok=True)
        prf.P2RANK_OUT_ROOT = p2rank_out
        prf.WORK_ROOT = WORK_ROOT / "p2rank_rf"
        prf.WORK_ROOT.mkdir(parents=True, exist_ok=True)

        has_pred = any(p2rank_out.rglob(f"{case}.pdb_predictions.csv"))
        if not has_pred:
            self.log(self.tr("log_p2rank_detect", case))
            prb.WORK_ROOT = WORK_ROOT / "p2rank_work"
            prb.WORK_ROOT.mkdir(parents=True, exist_ok=True)
            try:
                _, info = prb.run_one((case, str(receptor_pdb), str(p2rank_out)))
            except Exception as e:
                self.log(self.tr("log_p2rank_err", case, str(e))); return False, None, None
            if info is None: return False, None, None

        tmp_exp = case_out / "_tmp_exp" / case
        tmp_exp.mkdir(parents=True, exist_ok=True)
        if not (tmp_exp / "receptor.pdb").is_file(): shutil.copy2(receptor_pdb, tmp_exp / "receptor.pdb")

        if ligand_dir.name == case:
            lig_parent = ligand_dir.parent
        else:
            tmp_lig = case_out / "_tmp_lig" / case
            tmp_lig.mkdir(parents=True, exist_ok=True)
            for lig in ligand_dir.iterdir():
                if lig.is_file() and lig.suffix.lower() == ".pdb":
                    dst = tmp_lig / lig.name
                    if not dst.is_file(): shutil.copy2(lig, dst)
            lig_parent = tmp_lig.parent

        try:
            _, center, info = prf.process_one(case, str(tmp_exp.parent), str(lig_parent), str(case_out))
        except Exception as e:
            self.log(self.tr("log_p2rank_guide_err", case, str(e))); return False, None, None

        if center is None:
            self.log(self.tr("log_p2rank_fail", case, str(info))); return False, None, None

        src_box = case_out / f"p2rank_rf_{case}.box"
        dst_box = case_out / "receptor.box"
        box_size = cfg.get("box_size", 25.0)
        if src_box.is_file(): shutil.copy(src_box, dst_box)
        else:
            with open(dst_box, "w") as f:
                f.write(f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f} {box_size:.3f} {box_size:.3f} {box_size:.3f}\n")
        return True, tuple(center), (box_size, box_size, box_size)

    def _task_run_external_script(self, script_name: str):
        if self.is_running: return
        def run():
            self._start_task()
            script = _ROOT / "NEW" / script_name
            if not script.is_file():
                self.log(self.tr("log_no_script", str(script)))
                self._finish_task(); return
            env = os.environ.copy()
            env["VINA_BIN"] = self.vina_var.get()
            env["OBABEL_BIN"] = self.obabel_var.get()
            try:
                process = subprocess.Popen(
                    [sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", env=env
                )
                if process.stdout:
                    for line in process.stdout:
                        if self._stop_event.is_set():
                            process.terminate(); self.log(self.tr("log_proc_killed")); break
                        self.log(line.strip())
                process.wait()
            except Exception as e:
                self.log(self.tr("log_exec_err", str(e)))
            self._finish_task()
        threading.Thread(target=run, daemon=True).start()

if __name__ == "__main__":
    app = AF3MasterApp()
    app.mainloop()