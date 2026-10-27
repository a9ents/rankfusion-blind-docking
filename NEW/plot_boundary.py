import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= 顶刊全局样式 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['pdf.fonttype'] = 42  # 确保期刊出版时字体正确嵌入
plt.rcParams['axes.labelweight'] = 'bold' # 轴标签加粗

def plot_intervention_boundary():
    # 1. 准备数据
    data = [
        {"Method": "Original AF3", "Order": 1, "Probe_Radius": 0.14, "Delta_RMSD": 0.0},
        {"Method": "OpenMM Minim.", "Order": 2, "Probe_Radius": 0.11, "Delta_RMSD": 0.1},
        {"Method": "DLPacker", "Order": 3, "Probe_Radius": 0.15, "Delta_RMSD": 0.2},
        {"Method": "Thermal Anneal", "Order": 4, "Probe_Radius": 1.65, "Delta_RMSD": 0.87},
        {"Method": "SMD Pull", "Order": 5, "Probe_Radius": 2.76, "Delta_RMSD": 1.50},
        {"Method": "Magnetic/Drill", "Order": 6, "Probe_Radius": 0.24, "Delta_RMSD": 0.50},
        {"Method": "Ligand Insert", "Order": 7, "Probe_Radius": 1.72, "Delta_RMSD": 2.50},
        {"Method": "Relax (After Pull)", "Order": 8, "Probe_Radius": 0.55, "Delta_RMSD": 0.50} 
    ]
    df = pd.DataFrame(data)

    # 2. 建立画布
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    # 添加浅灰色细虚线网格
    ax.grid(True, linestyle='--', linewidth=0.6, color='gray', alpha=0.5, zorder=0)
    
    # 3. 绘制阈值参考线 (1.5 Å)
    ax.axhline(y=1.5, color='#555555', linestyle='-.', linewidth=1.5, zorder=1)
    # 将阈值文字放到右侧空白处，不遮挡数据
    ax.text(7.2, 1.55, "Water Molecule\nThreshold (1.5 Å)", 
            color='#333333', fontsize=10, fontweight='bold', ha='center', va='bottom')

    # 4. 绘制气泡散点图 (使用 seaborn 的 vlag 柔和渐变)
    scatter = sns.scatterplot(
        data=df, 
        x="Order", 
        y="Probe_Radius", 
        size="Delta_RMSD", 
        hue="Delta_RMSD",
        palette="vlag",      # 顶刊常用柔和红蓝渐变
        sizes=(100, 600),    # 气泡大小范围
        alpha=0.85,
        edgecolor="black",
        linewidth=1.0,
        ax=ax,
        zorder=2
    )

    # 5. 标注每个数据点的名称 (智能防重叠)
    for i in range(len(df)):
        row = df.iloc[i]
        method = row["Method"]
        
        # 预设默认偏移
        y_offset = 18
        x_offset = 0
        
        # 针对性微调，防止文字打架
        if method in ["Original AF3", "OpenMM Minim."]:
            y_offset = 22
        elif method in ["DLPacker", "Magnetic/Drill"]:
            y_offset = -25 # 放下面
        elif method == "SMD Pull":
            x_offset = -10
        elif method == "Ligand Insert":
            y_offset = 25
            
        # 加粗关键节点
        fw = 'bold' if method in ["Original AF3", "SMD Pull", "Ligand Insert"] else 'normal'
        
        ax.annotate(
            method, 
            (row["Order"], row["Probe_Radius"]),
            xytext=(x_offset, y_offset), 
            textcoords='offset points',
            ha='center', 
            fontsize=9,
            fontweight=fw,
            zorder=3
        )

    # 6. 美化图表框架
    ax.set_xticks(df["Order"])
    # 优化X轴标签描述
    ax.set_xticklabels(["Baseline", "Physics Min.", "AI Predict", "Thermal", "Mech. Pull", "Extreme", "Induced", "Relaxed"], 
                       rotation=45, ha='right', fontsize=10)
    
    ax.set_ylabel("Final Probe Radius (Å)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlabel("Downstream Intervention Strategy (Increasing Non-physical Force)", fontsize=12, fontweight='bold', labelpad=10)
    
    # 隐藏顶部和右侧边框，更清爽
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 刻度朝内
    ax.tick_params(axis='both', which='both', direction='in', length=4, width=1.2)

    # 调整图例 (Colorbar)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        
    norm = plt.Normalize(df["Delta_RMSD"].min(), df["Delta_RMSD"].max())
    sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)
    sm.set_array([])
    # 控制 Colorbar 高度和主图齐平
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Structural Damage (Δ Global RMSD, Å)", fontsize=11, fontweight='bold')
    cbar.ax.tick_params(direction='in', length=3)

    # 阴影标注区域
    ax.axvspan(3.5, 7.5, color='gray', alpha=0.08, zorder=1)
    
    plt.tight_layout()
    
    # 7. 保存图片 (600 dpi PNG + 矢量 PDF)
    plt.savefig("Intervention_Cost_Benefit.png", dpi=600, bbox_inches='tight')
    plt.savefig("Intervention_Cost_Benefit.pdf", bbox_inches='tight')
    print("图表已保存为 'Intervention_Cost_Benefit.png' 和 '.pdf'")
    plt.show()

# 运行函数
plot_intervention_boundary()