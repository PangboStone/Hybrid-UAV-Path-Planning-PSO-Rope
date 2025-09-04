import numpy as np
from calFitness import cost_function

def init_particles(self):
    """ 初始化粒子群 """
    particles = []

    # 获取起点和终点坐标
    start = np.array(self.drone.start_pos)
    end = np.array(self.drone.target_pos)

    for _ in range(self.nPop):
        # 在起点和终点之间生成均匀分布的节点
        t = np.linspace(0, 1, self.node)[:, np.newaxis]  # 生成 node 个均匀分布的参数
        # 生成直线上的基础位置
        base_position = start + t * (end - start)
        # 添加少量随机扰动，避免所有粒子完全相同
        random_offset = np.random.normal(0, 5, (self.node, 3))  # 使用正态分布生成扰动
        position = base_position + random_offset

        # 确保位置在有效范围内
        position = np.clip(position, 0, self.maprows)  # 假设地图大小为 200x200
        position = np.round(position).astype(int)  # 转换为整数坐标

        # 初始化较小的速度
        velocity = np.random.uniform(-2, 2, (self.node, 3))  # 使用更小的速度范围

        # 计算初始成本
        _, _, cost = cost_function(self, [position])

        particles.append({
            'position': [position],
            'velocity': [velocity],
            'cost': cost,
            'best_position': [position].copy(),
            'best_cost': cost
        })

        # 更新全局最优
        if cost < self.global_best['cost']:
            self.global_best['position'] = [position].copy()
            self.global_best['cost'] = cost

    return particles