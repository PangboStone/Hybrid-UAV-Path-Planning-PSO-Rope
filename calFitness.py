import numpy as np
from scipy.interpolate import CubicSpline


def cost_function(self, path):
    """ 计算成本，即路径长度和障碍物碰撞惩罚 """

    # 起点和终点
    start_pos = self.drone.start_pos  # 起点 (x, y, z)
    target_pos = self.drone.target_pos  # 终点 (x, y, z)
    #障碍物信息
    obstacle_interp = self.obstacles_interp
    height_map = self.height_map
    # 遍历所有路径的中间点
    for p in path:
        # 整合路径点
        x_seq = [start_pos[0]] + [pos[0] for pos in p] + [target_pos[0]]
        y_seq = [start_pos[1]] + [pos[1] for pos in p] + [target_pos[1]]
        z_seq = [start_pos[2]] + [pos[2] for pos in p] + [target_pos[2]]

    # 三次样条插值求路径
    full_path = interpolate_path(x_seq, y_seq, z_seq)

    # 检测路径是否与障碍冲突
    path_collisions = is_collision(full_path, obstacle_interp,height_map,self.safe_height)

    path_length = round(np.sum(np.linalg.norm(np.diff(full_path, axis=0), axis=1)),1)
    cost = path_length + path_collisions * 1000  # Penalty for collisions

    # # 计算路径长度
    # dx = np.diff(X_seq)
    # dy = np.diff(Y_seq)
    # dz = np.diff(Z_seq)
    # # 精度取小数点后3位置
    # path_length = round(np.sum(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)),3)

    return path_length, full_path, cost





def is_collision(full_path, obstacle_interp, height_map, safe_height):
    # 检测路径是否与障碍物相交
    collision_count = 0
    rows, cols = height_map.shape[:2]  # 获取 height_map 的行数和列数
    for point in full_path:
        x, y, z = point
        # obstacle_height = obstacle_interp((x, y))
        # if 0 <= x <200 and 0 <= y < 200 and z >= 0:
        if 0 <= x < rows and 0 <= y < cols and z >= 0:
            try:
                if z < height_map[y,x]+safe_height: # heatmap 的访问顺序不一样； 5是安全距离
                    collision_count += 1
            except IndexError:
                collision_count += 1
        else:
            collision_count += 1
    return collision_count

def is_collision_new(full_path, obstacle_interp, height_map):
    # 检测路径是否与障碍物相交
    collision_count = 0
    for point in full_path:
        x, y, z = point
        # 判断索引是否超出边界
        if 0 <= x < 200 and 0 <= y < 200 and z >= 0:
            try:
                obheight = height_map[x,y]
                if z < obheight:
                    collision_count += 1
            except IndexError:
                collision_count += 1
        else:
            collision_count += 1
    return collision_count

    #     if x < 0 or x >= 200 or y < 0 or y >= 200:
    #         # 超出边界视为碰撞
    #         collision_count += 1
    #         continue
    #
    #     height = obstacle_interp((x, y))
    #     obstacle_height = height_map[x,y]
    #
    #     if z < obstacle_height:
    #         collision_count += 1
    #
    #     if height is not None and z < height:
    #         collision_count += 1
    #
    # return collision_count


# 路径插值函数
def interpolate_path(x_seq, y_seq, z_seq):
    """
    Perform cubic spline interpolation on the path sequences.
    """
    # 归一化的原始索引序列，用于定义样条插值的输入节点
    i_seq = np.linspace(0, 1, len(x_seq))
    # 目标插值索引，生成100个等间隔的插值点
    I_seq = np.linspace(0, 1, 100)

    X_seq = np.round(CubicSpline(i_seq, x_seq)(I_seq)).astype(int)
    Y_seq = np.round(CubicSpline(i_seq, y_seq)(I_seq)).astype(int)
    Z_seq = np.round(CubicSpline(i_seq, z_seq)(I_seq)).astype(int)
    # 将 X, Y, Z 序列堆叠成形状为 (100, 3) 的矩阵，并返回
    return np.vstack((X_seq, Y_seq, Z_seq)).T


def function_test():
    global start_pos, target_pos, drone
    class Drone:
        def __init__(self, start_pos, target_pos):
            self.start_pos = start_pos
            self.target_pos = target_pos

    class Env:
        def __init__(self, drone, obstacle_interp):
            self.drone = drone
            self.obstacles_interp = obstacle_interp
            self.height_map = None
    # 生成起点和终点
    start_pos = (0, 0, 5)  # 起点坐标 (x, y, z)
    target_pos = (10, 10, 8)  # 终点坐标 (x, y, z)
    path = [
        [(2, 2, 6), (4, 4, 7), (6, 6, 6), (8, 8, 6)],  # 无碰撞路径
        [(2, 2, 3), (4, 5, 4), (6, 6, 3), (8, 9, 4)],  # 碰撞路径
        [(1, 3, 4), (3, 5, 3), (5, 6, 2), (8, 7, 3)]  # 碰撞路径
    ]


    # # 构建障碍物的网格数据
    # x = np.linspace(0, 10, 50)  # x 轴
    # y = np.linspace(0, 10, 50)  # y 轴
    # X, Y = np.meshgrid(x, y)  # 构建二维网格
    # Z = np.sin(X) + np.cos(Y) + 3  # 生成障碍物高度数据
    # # 创建一个插值器来模拟障碍物高度
    # from scipy.interpolate import RegularGridInterpolator
    # obstacle_interp = RegularGridInterpolator(
    #     (x, y), Z, bounds_error=False, fill_value=None
    # )


    # 或者用真实数据验证
    from defMap import process_map
    filepath = "testmap.tiff"
    X, Y, Z, obstacle_interp,_ = process_map(filepath, scale=1, grid_resolution=200)  # "Z"是障碍物的三维网格


    # 验证代码
    drone = Drone(start_pos, target_pos)
    env = Env(drone, obstacle_interp )
    # 3. 可视化设置
    import pyvista as pv
    plotter = pv.Plotter()
    terrain = pv.StructuredGrid(X, Y, Z)
    terrain_mesh = terrain.extract_surface()
    plotter.add_mesh(terrain_mesh, color='tan', show_edges=False, opacity=0.5, label='Terrain')
    colors = ['blue', 'red', 'green']
    # 遍历路径并计算成本
    for i, p in enumerate(path):
        path_label, full_path ,cost = cost_function(env, [p])
        print(f"{path_label}, 路径 {i + 1} 的成本为: {cost}")

        # 可视化路径
        plotter.add_lines(full_path, color=colors[i], width=3, label=f'Path {i + 1} (Cost: {cost:.2f})')
        plotter.add_mesh(pv.PolyData(full_path), color=colors[i], point_size=10, render_points_as_spheres=True)
    # 起点和终点
    plotter.add_mesh(pv.PolyData([start_pos]), color='yellow', point_size=20, render_points_as_spheres=True,
                     label='Start')
    plotter.add_mesh(pv.PolyData([target_pos]), color='cyan', point_size=20, render_points_as_spheres=True,
                     label='Target')
    # 显示
    plotter.add_legend()
    plotter.show()
if __name__ == "__main__":
    function_test()




