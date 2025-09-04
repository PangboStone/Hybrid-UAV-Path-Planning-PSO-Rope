import numpy as np
from scipy.interpolate import CubicSpline


class ElasticPathOptimizer:
    def __init__(self, tension=0.5, smoothness=0.3, step_size=0.1):
        """
        初始化弹性路径优化器
        :param tension: 张力系数，控制路径的收缩程度
        :param smoothness: 平滑系数，控制路径的光滑程度
        :param step_size: 节点移动步长
        """
        self.tension = tension
        self.smoothness = smoothness
        self.step_size = step_size

    def calculate_forces(self, path):
        """计算作用在每个节点上的力"""
        forces = np.zeros_like(path)
        n_points = len(path)

        for i in range(n_points):
            # 1. 张力（与相邻节点的吸引力）
            if i > 0:  # 与前一个节点的张力
                forces[i] += self.tension * (path[i - 1] - path[i])
            if i < n_points - 1:  # 与后一个节点的张力
                forces[i] += self.tension * (path[i + 1] - path[i])

            # 2. 平滑力（使路径更平滑）
            if 0 < i < n_points - 1:
                prev_vec = path[i] - path[i - 1]
                next_vec = path[i + 1] - path[i]
                smooth_force = self.smoothness * (prev_vec + next_vec)
                forces[i] += smooth_force

        return forces

    def optimize_path(self, path, height_map, max_iterations=100, convergence_threshold=0.01):
        """
        优化路径
        :param path: 初始路径
        :param height_map: 高度图，用于碰撞检测
        :return: 优化后的路径
        """
        current_path = np.array(path)
        prev_cost = float('inf')

        for iteration in range(max_iterations):
            # 计算作用力
            forces = self.calculate_forces(current_path)

            # 更新节点位置
            new_path = current_path + self.step_size * forces

            # 确保路径不会穿过障碍物
            new_path = self.adjust_for_obstacles(new_path, height_map)

            # 计算新路径的成本
            current_cost = self.calculate_path_cost(new_path)

            # 检查收敛性
            if abs(current_cost - prev_cost) < convergence_threshold:
                break

            current_path = new_path
            prev_cost = current_cost

        return current_path

    def adjust_for_obstacles(self, path, height_map):
        """调整路径避免碰撞"""
        adjusted_path = path.copy()

        for i in range(len(path)):
            x, y, z = path[i]
            x, y = int(x), int(y)

            if 0 <= x < height_map.shape[1] and 0 <= y < height_map.shape[0]:
                obstacle_height = height_map[y, x]
                if z < obstacle_height:
                    # 如果发生碰撞，向上调整高度
                    adjusted_path[i, 2] = obstacle_height + 1

        return adjusted_path

    def calculate_path_cost(self, path):
        """计算路径成本（长度）"""
        return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
