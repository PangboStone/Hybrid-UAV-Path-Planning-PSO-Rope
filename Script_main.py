from defMap import process_map
from PSO import PSO, Drone
from calFitness import cost_function
import pyvista as pv
from map_selector import MapSelector
import numpy as np

if __name__ == "__main__":
    # start_pos = (5, 5, 5)  # 无人机起始位置
    # target_pos = (150, 50, 170)  # 目标位置

    # 生成障碍物信息
    filepath = "map/Mountainous Terrain.tiff"
    terrainfile = "3D Env/Mountainous.vtk"
    # 设定起始点和目标点
    selector = MapSelector(filepath)
    print("Constructing Environment")
    start_pos, target_pos = selector.select_points()
    X, Y, Z, obstacles_interp, height_map = process_map(filepath)
    print("Grid Map Constructed")


    drone = Drone(start_pos, target_pos)
    print("Initializing Particles")
    pso = PSO(drone, nPop=50, MaxIt=500, node=3, w=1, c1=1.2, c2=3, obstacles_interp=obstacles_interp,height_map = height_map)
    pso.interator()
    best_path = pso.global_best['position']
    best_distance= pso.global_best['cost']

    """
    路径信息要转化为：
    [(2, 2, 6), (4, 4, 7), (6, 6, 6), (8, 8, 6)]
    的格式才能进行绘制

    还有为什么修改节点数会导致报错，待查一下
    """

    _,full_path,_ = cost_function(pso, pso.global_best['position'])

    for point in full_path:
        x, y, z= point
        height = pso.obstacles_interp((x, y))
        # print(f"路径点: ({x}, {y}),路径高度：{z}, 障碍物网格高度: {height}, 实际真实高度{pso.height_map[y,x]}")


    # 绘制环境
    plotter = pv.Plotter()
    terrain = pv.StructuredGrid(X, Y, Z)
    # terrain_mesh = pv.read(terrainfile)
    terrain_mesh = terrain.extract_surface()
    plotter.add_mesh(terrain_mesh, color='white', show_edges=False)

    # 添加光影效果
    light = pv.Light()
    light.set_direction_angle(20, -20)
    plotter.add_light(light)
    light.intensity = 0.1

    # 可视化路径
    plotter.add_lines(full_path, color='red', width=6, label=f'Best Path (Cost: {best_distance:.2f})',connected=True)

    plotter.add_mesh(pv.PolyData(full_path), color='lime', point_size=10, render_points_as_spheres=True)

    # 起点和终点
    plotter.add_mesh(pv.PolyData([start_pos]), color='yellow', point_size=20, render_points_as_spheres=True, label='Start')
    plotter.add_mesh(pv.PolyData([target_pos]), color='cyan', point_size=20, render_points_as_spheres=True, label='Target')
    plotter.background_color = 'white'
    plotter.camera_position = 'xz'
    plotter.camera.elevation = 30
    plotter.camera.azimuth = 45
    # 显示
    plotter.add_legend()
    plotter.show()



"""
1. 算法优化
2. 找对比论文（简单点的，找优缺点，优化弱项）
3. 找对比指标
4. 找常用的经典算法的参考文献


论文时间： 5月，英文
"""