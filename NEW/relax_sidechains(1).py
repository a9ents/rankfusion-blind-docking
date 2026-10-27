import sys
from openmm.app import PDBFile, Modeller, ForceField, Simulation, NoCutoff, HBonds
from openmm import LangevinMiddleIntegrator, Platform, unit, CustomExternalForce

def relax_sidechains(input_pdb, output_pdb):
    print(f"正在读取结构: {input_pdb}")
    pdb = PDBFile(input_pdb)
    modeller = Modeller(pdb.topology, pdb.positions)

    print("正在添加氢原子...")
    modeller.addHydrogens(pH=7.0)

    print("正在加载 Amber14 力场...")
    forcefield = ForceField('amber14-all.xml') # 局部优化不需要水模型

    # 创建系统（无周期性边界，模拟真空/隐式环境，极快）
    system = forcefield.createSystem(modeller.topology, 
                                     nonbondedMethod=NoCutoff, 
                                     constraints=HBonds)

    # 限制主链（C, CA, N, O），让侧链自由运动
    print("正在施加主链位置限制...")
    restraint = CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter('k', 100.0 * unit.kilocalories_per_mole / unit.angstroms**2) 
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    for atom in modeller.topology.atoms():
        if atom.name in ['CA', 'C', 'N', 'O']:
            pos = modeller.positions[atom.index]
            restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])

    system.addForce(restraint)

    integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    try:
        platform = Platform.getPlatformByName('CUDA')
    except Exception:
        print("未检测到 CUDA，使用 CPU 运行...")
        platform = Platform.getPlatformByName('CPU')

    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print("正在进行能量最小化 (Energy Minimization)...")
    simulation.minimizeEnergy(maxIterations=2000)

    print(f"正在保存精修后的结构至: {output_pdb}")
    positions = simulation.context.getState(getPositions=True).getPositions()
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(simulation.topology, positions, f)
    
    print("✅ 侧链精修完成！")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python relax_sidechains.py <input.pdb> <output.pdb>")
        sys.exit(1)
    
    relax_sidechains(sys.argv[1], sys.argv[2])