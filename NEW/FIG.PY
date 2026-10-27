# 1. 环境与基础设置
reinitialize
cd E:\AF3环境\ALL-DATA\SMD_REPAIR_RESULTS
bg_color white
set ray_opaque_background, on
set ray_trace_mode, 1         # 启用轮廓线（让结构边缘更清晰）
set ray_trace_color, black
set ambient, 0.4
set specular, 0.2
set cartoon_transparency, 0.5
set sphere_transparency, 0.3
set stick_radius, 0.15
set label_size, 24
set label_color, black
set label_outline_color, white
set label_position, (0, 0, 2)

# 2. 循环处理四个 case
python
cases = ["6XBO_5MC", "7BLA_WCS", "7MY1_IPE", "7N4W_P4V"]
for c in cases:
    cmd.reinitialize()
    cmd.bg_color("white")
    
    # 重新设置渲染参数 (因为 reinitialize 会重置)
    cmd.set("ray_opaque_background", "on")
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_color", "black")
    cmd.set("ambient", 0.4)
    cmd.set("specular", 0.2)
    cmd.set("cartoon_transparency", 0.5)
    cmd.set("sphere_transparency", 0.3)
    cmd.set("stick_radius", 0.15)
    cmd.set("label_size", 24)
    cmd.set("label_color", "black")
    cmd.set("label_outline_color", "white")
    cmd.set("label_position", (0, 0, 2))
    
    # 载入结构
    cmd.load(f"{c}_apo.pdb", "apo")
    cmd.load(f"{c}_opened.pdb", "opened")
    
    # 证明重合：opened 对齐到 apo
    rms = cmd.align("opened", "apo")
    print(f"[{c}] APO vs OPENED RMSD: {rms[0]:.4f} A")
    
    # 主体渲染：apo 灰色 Cartoon，opened 蓝色线框
    cmd.show("cartoon", "apo")
    cmd.color("gray70", "apo")
    
    cmd.show("lines", "opened")
    cmd.color("marine", "opened")
    cmd.set("line_width", 2.0, "opened")
    
    # 找出异常 Loop 区 (黄色)
    cmd.color("yellow", "apo and ss L+")
    
    # 找出严重 Clash 区域 (红色球体)
    # 选取非骨架重原子，且与周围其他原子距离极近的
    cmd.select("clash_atoms", "apo and not (name C+N+O+CA) and (all within 1.6 of apo)")
    cmd.show("spheres", "clash_atoms")
    cmd.color("red", "clash_atoms")
    cmd.set("sphere_scale", 0.3, "clash_atoms")
    
    # 寻找口袋中心位置进行标注
    # 方法：用配体文件如果存在的话，或者用所选 Clash 原子的几何中心
    # 这里我们用一个虚拟球体标记一个泛指的空腔或配体位置
    if cmd.count_atoms("clash_atoms") > 0:
        center = cmd.get_position("clash_atoms")
        cmd.pseudoatom("pocket_center", pos=center)
        cmd.show("spheres", "pocket_center")
        cmd.color("orange", "pocket_center")
        cmd.set("sphere_scale", 1.5, "pocket_center")
        
        # 添加标签
        cmd.label("pocket_center", f'"{c} - Blocked Center"')
    
    # 对最严重的 clashing 残基进行标注
    cmd.select("worst_clash", "byres clash_atoms")
    # 选择距离 pocket_center 最近的 3 个残基标注
    cmd.label("worst_clash and name CA and (byres pocket_center within 8)", "'%s%s' % (resn, resi)")
    
    # 视角设置
    cmd.orient("apo")
    cmd.zoom("apo", buffer=3)
    
    # 渲染并保存高分辨率图 (2400x1800 为 600dpi 下 4x3 英寸)
    out_file = f"{c}_diagnostic_600dpi.png"
    cmd.png(out_file, width=2400, height=1800, dpi=600, ray=1)
    print(f"✅ 已保存: {out_file}")
python end