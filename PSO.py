import numpy as np
from defMap import process_map
from iteratePSO import Iterate
from defParticle import init_particles


class Drone:
    def __init__(self, start_pos, target_pos):
        self.start_pos = start_pos
        self.target_pos = target_pos


class PSO:
    def __init__(self, drone, nPop, MaxIt,node, w, c1, c2, obstacles_interp, height_map):
        self.drone = drone  # Drone 对象，包含始末点信息
        self.nPop = nPop    # 种群/粒子数
        self.MaxIt = MaxIt  # 迭代次数
        self.node = node    # 路径插值节点数
        #   PSO 更新参数
        self.w = w
        self.c1 = c1
        self.c2 = c2
        #   障碍物信息
        self.obstacles_interp = obstacles_interp
        self.height_map = height_map
        self.safe_height = 10 #默认安全距离
        self.maprows, self.mapcols = height_map.shape[:2]
        #   路径信息
        self.global_best = {'position': None, 'cost': float('inf')}
        #   初始化粒子群
        self.particles = init_particles(self)


    def interator(self):
        #   迭代函数
        print(f"初始化结果{self.global_best}")
        Iterate(self)


if __name__ == "__main__":
    start_pos = (0, 0, 0)  # 无人机起始位置
    target_pos = (100, 100, 80)  # 目标位置

    # 生成障碍物信息
    filepath = "testmap.tiff"

    X, Y, Z, obstacles_interp, height_map = process_map(filepath)

    # 全局设定小数点保留位数为3
    np.set_printoptions(precision=3)

    drone = Drone(start_pos, target_pos)
    pso = PSO(drone, nPop=50, MaxIt=200, node=2, w=1, c1=1.5, c2=1.5, obstacles_interp=obstacles_interp, height_map = height_map)
    print('Construct Success')
    print(pso.global_best)

    # from calFitness import cost_function
    # _, full_path, _ = cost_function(pso, pso.global_best['position'])
    # print(type(full_path))
    # print(full_path.shape)
    # print(full_path[:5])
