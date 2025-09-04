import numpy as np
from PIL import Image
from scipy.interpolate import griddata, RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import pyvista as pv
import rasterio


def process_map(filepath, scale=1.0, grid_resolution=400):
    """
    处理二维格栅图，生成三维网格数据。
    :param filepath: 格栅图文件路径
    :param scale: 灰度值到高度的比例系数
    :param grid_resolution: 输出网格分辨率
    :return: 三维网格的 X, Y, Z 坐标
    """
    # 读取格栅图
    img = Image.open(filepath)
    # 获取RGB通道中的一个（比如Red通道）
    grid = np.array(img)[:, :, 0]  # 只取红色通道
    rows, cols = grid.shape

    # 创建网格的顶点坐标
    # 创建网格坐标
    x = np.linspace(0, cols - 1, grid_resolution)
    y = np.linspace(0, rows - 1, grid_resolution)
    X, Y = np.meshgrid(x, y)

    # 地面与建筑物分离 # 其实这一步也可以省去
    ground_level = np.min(grid)
    # height_map = (255-grid) * scale
    # height_map[grid == 255] = ground_level  # 地面高度设为 0
    height_map = grid * scale
    # height_map[grid == 0] = ground_level  # 地面高度设为 0


    # 生成建筑物点云
    points = []
    values = []
    for r in range(rows):
        for c in range(cols):
            points.append([c, r])  # 网格点位置
            values.append(height_map[r, c])  # 对应高度值
    points = np.array(points)
    values = np.array(values)

    # 插值生成三维网格, 采用线性插值，计算量较小
    Z = griddata(points, values, (X, Y), method="linear", fill_value=ground_level)

    # 进一步插值，提高精度，满足曲线的查询条件
    # 使用 RegularGridInterpolator 进行障碍物高度插值
    obstacle_interp = RegularGridInterpolator((x,y), Z, bounds_error=False, fill_value=None)

    return X, Y, Z, obstacle_interp, height_map

def visualize_3d(X, Y, Z):
    """
    可视化生成的三维网格。
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title("3D Map Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    plt.gca().invert_yaxis()
    plt.show()

def visualize_pyvista(x, y, z):
    """
    可视化三维网格数据的函数
    :param x: x 坐标数组
    :param y: y 坐标数组
    :param z: z 坐标数组
    """
    # 创建 PyVista 网格
    terrain_mesh = pv.StructuredGrid(x, y, z)
    # 将 z 坐标设置为网格的点数据，用于着色
    # terrain_mesh.point_data["height"] = z.flatten(order='F')  # 使用 Fortran order 展开以匹配 PyVista 的默认顺序
    # 可视化网格
    plotter = pv.Plotter()
    plotter.add_mesh(terrain_mesh, cmap='terrain',show_edges=True)
    # plotter.add_scalar_bar(title="Height") # 添加颜色条以解释颜色
    plotter.show()
    terrain_mesh.save('Mountainous.vtk')

# # 使用示例
# filepath = "map/Mountainous Terrain.tiff"
# X, Y, Z, obstacle_interp,_ = process_map(filepath, scale=1, grid_resolution=400)   #"Z"是障碍物的三维网格
#
# # 可视化结果
# visualize_pyvista(X,Y,obstacle_interp((X, Y)))





# # 加载您的网格
# loaded_mesh = pv.read('3D Env/Urban.vtk')
#
# # 创建 Plotter 对象
# plotter = pv.Plotter()
# light = pv.Light()
# light.set_direction_angle(20, -20)
# plotter.add_light(light)
# light.intensity = 0.1
# # plotter.enable_shadows()
#
# plotter.add_mesh(loaded_mesh, color='white', show_edges=False)
# plotter.show()