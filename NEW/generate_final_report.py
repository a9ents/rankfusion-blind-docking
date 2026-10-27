# -*- coding: utf-8 -*-
import os
import pandas as pd

# 替换为你实际的 CSV 文件路径
FINAL_COMPARISON_CSV = r"E:\AF3环境\ALL-DATA\probe_final_comparison\final_comparison.csv"
VERIFICATION_CSV = r"E:\AF3环境\ALL-DATA\SMD_REPAIR_RESULTS\verification_summary.csv"
OUT_REPORT = r"E:\AF3环境\ALL-DATA\FINAL_SUMMARY_REPORT.md"

def generate_report():
    report = []
    report.append("# AF3 盲对接瓶颈与下游干预失效边界：全量数据综合诊断报告\n")
    report.append("## 一、 全局数据集分析 (实验 vs AF3)\n")
    
    # 1. 基础对比数据 (来自你的 final_comparison.csv)
    if os.path.exists(FINAL_COMPARISON_CSV):
        df = pd.read_csv(FINAL_COMPARISON_CSV)
        total_cases = len(df)
        # 假设你有 probe_af3, probe_openmm, probe_dlp 等列
        median_af3 = df['probe_af3'].median() if 'probe_af3' in df else "N/A"
        median_omm = df['probe_openmm'].median() if 'probe_openmm' in df else "N/A"
        
        report.append(f"- **测试样本总数**: {total_cases}")
        report.append(f"- **AF3 原始结构中位探针半径**: {median_af3:.2f} Å")
        report.append(f"- **OpenMM 干预后中位探针**: {median_omm:.2f} Å")
        report.append("- **核心现象**: AF3 在低 ipTM 下，真实结合位点被侧链与主链的拓扑畸变严重堵塞。\n")
    else:
        report.append("[缺失] 未找到基础对比 CSV。\n")

    report.append("## 二、 极限干预测试 (四层验证)\n")
    
    if os.path.exists(VERIFICATION_CSV):
        # 注意读取编码
        try:
            vdf = pd.read_csv(VERIFICATION_CSV, encoding='utf-8-sig')
        except:
            vdf = pd.read_csv(VERIFICATION_CSV, encoding='gbk')
            
        report.append("### 2.1 样本状态概览\n")
        report.append("| Case | 初始 Probe | 预处理 Rama | 几何 Clash | L1 回弹 | L2 骨架质量 |")
        report.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        
        for idx, row in vdf.iterrows():
            case = row.get('case', 'Unknown')
            seed_probe = row.get('seed_probe', 'NA')
            rama = row.get('filter_rama', 'NA')
            clash = row.get('geom_clash_before', 'NA')
            l1_verdict = row.get('L1_verdict', 'NA')
            l2_verdict = row.get('L2_verdict', 'NA')
            
            report.append(f"| {case} | {seed_probe} Å | {rama} | {clash} | {l1_verdict} | {l2_verdict} |")
            
        report.append("\n### 2.2 核心失效机制归纳\n")
        report.append("1. **输入层面崩塌**: 大多数死结样本在 AF3 预测的 apo 状态下，骨架 Rama 允许区比例极低（30%-40%），且存在上千个严重空间冲突。")
        report.append("2. **干预无法生效**: 配体插入与物理精修未能改变核心拓扑，Opened 结构与 Apo 结构的 RMSD 偏离 8-20 Å。")
        report.append("3. **热力学不稳定**: 撤去外力后，由于错误折叠的骨架物理占据了口袋空间，任何人为撑开的空腔在常温水中会迅速坍塌（L1/L3 数据验证）。")
        
    else:
        report.append("[缺失] 未找到干预验证结果 CSV。\n")

    report.append("\n## 三、 总结论与论文写作建议 (JCIM)\n")
    report.append("### 3.1 论文主线 (Storyline)")
    report.append("**不要写成'我们尝试了什么方法修口袋'，要写成'我们定量界定了盲对接下游干预的物理边界'。**")
    report.append("论点链：AF3结构质量下降 -> 瓶颈从排序变检测 -> 寻找原因 -> 发现主链拓扑刚性是死结 -> 尝试不同物理/AI手段 -> 证明除非结构解体，否则无法打破死结。")
    
    report.append("\n### 3.2 建议图表排列")
    report.append("- **Figure 1**: 漏斗图或散点图，展示从 424 个实验结构到 421 个 AF3 结构的瓶颈迁移（检测器失败）。")
    report.append("- **Figure 2**: 真实位点探针分布柱状图（0.14 Å vs 2.01 Å），以及 PyMOL 渲染的“满屏冲突与堵塞”高分辨率图。")
    report.append("- **Figure 3**: 干预代价-收益图（基于上面的 Python 画图代码），展示高 RMSD 恶化换取微小探针释放。")
    report.append("- **Table 1**: 四层热力学验证汇总（L1-L4）。")
    report.append("\n### 3.3 核心金句 (直接放 Abstract)")
    report.append("> \"Our results demonstrate that for low-confidence AF3 predictions, the failure is fundamentally an input-level structural collapse rather than an algorithmic deficiency. Any downstream pocket repair relies on external mechanical intervention, which inevitably leads to thermodynamic instability and topological destruction.\"")

    # 写入文件
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"✅ 综合诊断报告已生成: {OUT_REPORT}")

if __name__ == "__main__":
    generate_report()