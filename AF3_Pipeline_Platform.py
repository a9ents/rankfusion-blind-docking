# pyright: reportAttributeAccessIssue=false
#!/usr/bin/env python3
"""
AF3_Platform_App.py — AF3 结构与下游干预综合软件平台 (OOP 完整版)
整合：数据转换、DLPacker修复、Vina对接、探针分析、OpenMM物理干预
"""
import os
import sys
import json
import shutil
import threading
import traceback
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime

# 尝试导入底层库（如果用户环境未安装，延迟报错）
try:
    import numpy as np
    import gemmi
    import pandas as pd
except ImportError as e:
    print(f"警告：缺少核心依赖 {e}，部分高级功能将不可用。请确保安装了 gemmi, numpy, pandas。")

# =====================================================================
# 全局配置
# =====================================================================
_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = _ROOT / "platform_oop_config.json"
NEW_DIR = _ROOT / "NEW"
PIPELINE_DIR = _ROOT / "pipeline"

for _sub in (NEW_DIR, PIPELINE_DIR, _ROOT / "src"):
    if str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))


class PipelinePlatform(tk.Tk):
    """主程序类，继承自 tk.Tk，封装所有 self 属性和方法"""
    
    def __init__(self):
        super().__init__()
        self.title("AF3 综合流水线与物理干预平台 (OOP 版)")
        self.geometry("1100x850")
        
        # 状态变量
        self.config_data = self._load_config()
        self._stop_event = threading.Event()
        
        # 构建 UI
        self._build_ui()
        
    # =================================================================
    # 配置管理 (self 方法)
    # =================================================================
    def _load_config(self) -> dict:
        """加载本地配置文件"""
        if CONFIG_FILE.is_file():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_config(self):
        """自动保存当前所有路径配置"""
        cfg = {
            "vina": self.vina_var.get(),
            "obabel": self.obabel_var.get(),
            "root_dir": self.root_dir_var.get(),
            "p2rank": self.p2rank_var.get(),
            "wsl_env": self.wsl_env_var.get()
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.log(f"⚠ 配置保存失败: {e}")

    # =================================================================
    # UI 构建 (self 方法)
    # =================================================================
    def _build_ui(self):
        # 0. 顶部配置栏
        env_frame = ttk.LabelFrame(self, text="0. 全局环境配置 (修改后自动保存)", padding=10)
        env_frame.pack(fill="x", padx=10, pady=5)

        self.vina_var = tk.StringVar(value=self.config_data.get("vina", ""))
        self.obabel_var = tk.StringVar(value=self.config_data.get("obabel", ""))
        self.root_dir_var = tk.StringVar(value=self.config_data.get("root_dir", r"E:\AF3环境\ALL-DATA"))
        self.p2rank_var = tk.StringVar(value=self.config_data.get("p2rank", ""))
        self.wsl_env_var = tk.StringVar(value=self.config_data.get("wsl_env", "fpocket_env"))

        self._add_config_row(env_frame, 0, "Vina 程序:", self.vina_var, "file")
        self._add_config_row(env_frame, 1, "Obabel 程序:", self.obabel_var, "file")
        self._add_config_row(env_frame, 2, "项目主目录:", self.root_dir_var, "dir")
        self._add_config_row(env_frame, 3, "P2Rank 目录:", self.p2rank_var, "dir")

        # 选项卡
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        self.tab_convert = ttk.Frame(self.notebook, padding=10)
        self.tab_dock = ttk.Frame(self.notebook, padding=10)
        self.tab_probe = ttk.Frame(self.notebook, padding=10)
        self.tab_openmm = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_convert, text="1. 数据转换与修复")
        self.notebook.add(self.tab_dock, text="2. 分子对接 (Vina)")
        self.notebook.add(self.tab_probe, text="3. 探针与空腔分析")
        self.notebook.add(self.tab_openmm, text="4. 物理干预 (OpenMM)")

        self._build_tab_convert()
        self._build_tab_dock()
        self._build_tab_probe()
        self._build_tab_openmm()

        # 底部日志与控制
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill="x", padx=10, pady=5)
        self.btn_stop = ttk.Button(ctrl_frame, text="⏹ 停止所有任务", command=self._stop_tasks, state="disabled")
        self.btn_stop.pack(side="left")
        
        log_frame = ttk.LabelFrame(self, text="运行日志", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_txt = scrolledtext.ScrolledText(log_frame, height=12, font=("Consolas", 9), wrap="word", bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
        self.log_txt.pack(fill="both", expand=True)

    def _add_config_row(self, parent, row, label, var, pick_type):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=80).grid(row=row, column=1, padx=6)
        btn = ttk.Button(parent, text="选择…", command=lambda: self._pick_path(var, pick_type))
        btn.grid(row=row, column=2)

    def _build_tab_convert(self):
        ttk.Label(self.tab_convert, text="将 AF3 的 CIF 转换为 PDB，并使用 DLPacker 重构侧链").pack(anchor="w", pady=10)
        ttk.Button(self.tab_convert, text="▶ 批量 CIF 转 PDB", command=self._task_cif_to_pdb).pack(anchor="w", pady=5)
        ttk.Button(self.tab_convert, text="▶ 批量 DLPacker 侧链重建", command=self._task_dlp_reconstruct).pack(anchor="w", pady=5)

    def _build_tab_dock(self):
        row1 = ttk.Frame(self.tab_dock)
        row1.pack(fill="x", pady=5)
        ttk.Label(row1, text="大分子目录:").pack(side="left")
        self.dock_rec_var = tk.StringVar(value=self.config_data.get("dock_rec", ""))
        ttk.Entry(row1, textvariable=self.dock_rec_var, width=70).pack(side="left", padx=5)
        ttk.Button(row1, text="选", command=lambda: self._pick_path(self.dock_rec_var, "dir")).pack(side="left")

        row2 = ttk.Frame(self.tab_dock)
        row2.pack(fill="x", pady=5)
        ttk.Label(row2, text="配体父目录:").pack(side="left")
        self.dock_lig_var = tk.StringVar(value=self.config_data.get("dock_lig", ""))
        ttk.Entry(row2, textvariable=self.dock_lig_var, width=70).pack(side="left", padx=5)
        ttk.Button(row2, text="选", command=lambda: self._pick_path(self.dock_lig_var, "dir")).pack(side="left")

        ttk.Button(self.tab_dock, text="▶ 运行盲对接 (Vina)", command=self._task_docking).pack(anchor="w", pady=15)

    def _build_tab_probe(self):
        ttk.Label(self.tab_probe, text="计算真值配体位置处的空腔探针半径，并对比不同结构").pack(anchor="w", pady=10)
        ttk.Button(self.tab_probe, text="▶ 全量探针对比 (AF3 vs OpenMM vs DLP)", command=self._task_compare_all_probes).pack(anchor="w", pady=5)
        ttk.Button(self.tab_probe, text="▶ 分析奇迹生还案例", command=self._task_analyze_survivors).pack(anchor="w", pady=5)

    def _build_tab_openmm(self):
        ttk.Label(self.tab_openmm, text="利用 OpenMM 进行侧链修复、冷热循环退火或配体引导插入").pack(anchor="w", pady=10)
        ttk.Button(self.tab_openmm, text="▶ 冷冻-回暖循环 (cold.py)", command=lambda: self._task_run_external_script("cold.py")).pack(anchor="w", pady=5)
        ttk.Button(self.tab_openmm, text="▶ 渐进式物理牵引 (ligand_fit.py)", command=lambda: self._task_run_external_script("ligand_fit.py")).pack(anchor="w", pady=5)
        ttk.Button(self.tab_openmm, text="▶ 四层热力学验证 (every case.py)", command=lambda: self._task_run_external_script("every case.py")).pack(anchor="w", pady=5)

    # =================================================================
    # 通用工具方法 (self 方法)
    # =================================================================
    def _pick_path(self, var, path_type):
        if path_type == "dir":
            p = filedialog.askdirectory()
        else:
            p = filedialog.askopenfilename(filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if p:
            var.set(p)
            self._save_config()

    def log(self, msg):
        """安全写日志（跨线程）"""
        def _append():
            self.log_txt.insert("end", msg + "\n")
            self.log_txt.see("end")
        self.after(0, _append)

    def _exec_thread(self, target_func, *args):
        """多线程执行器"""
        self._stop_event.clear()
        self.btn_stop.configure(state="normal")
        
        def wrapper():
            try:
                target_func(*args)
            except Exception as e:
                self.log(f"❌ 任务异常: {e}\n{traceback.format_exc()}")
            finally:
                self.after(0, lambda: self.btn_stop.configure(state="disabled"))
                self.log(f"========== 任务结束 {datetime.now().strftime('%H:%M:%S')} ==========")
                
        self.log(f"========== 开始新任务 {datetime.now().strftime('%H:%M:%S')} ==========")
        self._save_config()
        threading.Thread(target=wrapper, daemon=True).start()

    def _stop_tasks(self):
        self._stop_event.set()
        self.log("⏹ 已发送停止信号...")

    # =================================================================
    # 具体业务逻辑 (self 方法)
    # =================================================================
    def _task_cif_to_pdb(self):
        def run():
            root = Path(self.root_dir_var.get())
            src = root / "DATA" / "AF3_workflow"
            dst = root / "DATA" / "AF3-PBD"
            dst.mkdir(parents=True, exist_ok=True)
            
            cifs = list(src.glob("*/*_MODEL.CIF")) + list(src.glob("*/*.cif"))
            self.log(f"找到 {len(cifs)} 个 CIF 文件，开始转换...")
            
            for cif in cifs:
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
                    self.log(f"  ✓ {case}")
                except Exception as e:
                    self.log(f"  ✗ {case} 失败: {e}")
        self._exec_thread(run)

    def _task_dlp_reconstruct(self):
        def run():
            try:
                from dlpacker import DLPacker
            except ImportError:
                self.log("❌ 未安装 dlpacker，无法执行。")
                return
                
            root = Path(self.root_dir_var.get())
            src = root / "DATA" / "AF3-PBD"
            dst = root / "DATA" / "AF3-PBD-DLP"
            dst.mkdir(parents=True, exist_ok=True)
            
            pdbs = list(src.glob("*_apo.pdb"))
            self.log(f"找到 {len(pdbs)} 个 apo 结构，开始 DLPacker 重建...")
            for pdb in pdbs:
                case = pdb.stem.replace("_apo", "")
                out_pdb = dst / f"{case}_dlp.pdb"
                if out_pdb.exists(): continue
                try:
                    dlp = DLPacker(str(pdb))
                    dlp.reconstruct_protein(order='sequence', output_filename=str(out_pdb))
                    self.log(f"  ✓ {case}")
                except Exception as e:
                    self.log(f"  ✗ {case} 失败: {e}")
        self._exec_thread(run)

    def _task_docking(self):
        def run():
            rec_dir = Path(self.dock_rec_var.get())
            lig_dir = Path(self.dock_lig_var.get())
            if not rec_dir.is_dir() or not lig_dir.is_dir():
                self.log("❌ 路径无效！")
                return
                
            try:
                import auto_pipeline as ap
                ap.VINA_BIN = self.vina_var.get()
                ap.OBABEL_BIN = self.obabel_var.get()
                
                out_dir = self.root_dir_var.get() + "/DOCKING_RESULTS"
                Path(out_dir).mkdir(exist_ok=True)
                
                pdbs = list(rec_dir.glob("*.pdb"))
                self.log(f"开始对接 {len(pdbs)} 个大分子...")
                
                for p in pdbs:
                    if self._stop_event.is_set(): break
                    case = p.stem
                    case_lig = lig_dir / case
                    if not case_lig.is_dir(): continue
                    
                    self.log(f"对接 [{case}] ...")
                    ap.CFG["blind_mode"] = "pocket_guided"
                    # 这里调用底层 run_pocket_guided
                    ok, _ = ap.run_pocket_guided(str(p), str(case_lig), out_dir)
                    self.log(f"  {'✓ 成功' if ok else '✗ 失败'}")
            except Exception as e:
                self.log(f"对接异常: {e}")
        self._exec_thread(run)

    def _task_compare_all_probes(self):
        def run():
            self.log("正在执行全量探针对比（可能需要几分钟）...")
            script = NEW_DIR / "compare_all_probes(2).py"
            if not script.is_file():
                self.log(f"❌ 未找到脚本: {script}")
                return
            self._run_subprocess(script)
        self._exec_thread(run)

    def _task_analyze_survivors(self):
        def run():
            script = NEW_DIR / "analyze_survivors(1).py"
            if not script.is_file():
                self.log(f"❌ 未找到脚本: {script}")
                return
            self._run_subprocess(script)
        self._exec_thread(run)

    def _task_run_external_script(self, script_name):
        def run():
            script = NEW_DIR / script_name
            if not script.is_file():
                self.log(f"❌ 未找到脚本: {script}")
                return
            self.log(f"正在启动 {script_name} ...")
            self._run_subprocess(script)
        self._exec_thread(run)

    def _run_subprocess(self, script_path):
        """封装 subprocess 调用，实时输出日志"""
        env = os.environ.copy()
        env["VINA_BIN"] = self.vina_var.get()
        env["OBABEL_BIN"] = self.obabel_var.get()
        
        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=env
            )
            for line in process.stdout:
                if self._stop_event.is_set():
                    process.terminate()
                    self.log("⏹ 进程已终止")
                    break
                self.log(line.strip())
            process.wait()
        except Exception as e:
            self.log(f"执行异常: {e}")

# =====================================================================
# 启动入口
# =====================================================================
if __name__ == "__main__":
    app = PipelinePlatform()
    app.mainloop()