import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Generalized Rastrigin (F9) 函数
def rastrigin(x):
    """计算Generalized Rastrigin函数值"""
    return 10 * len(x) + sum([(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])


# 改进的粒子群优化类（带动态惯性权重）
class DiversityAwarePSO:
    def __init__(self, cost_func, dim, num_particles, max_iter, bounds):
        """
        初始化改进的PSO

        参数:
            cost_func: 目标函数
            dim: 问题维度
            num_particles: 粒子数量
            max_iter: 最大迭代次数
            bounds: 每个维度的边界 [(min, max), ...]
        """
        self.cost_func = cost_func
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds

        # PSO参数 - 初始值
        self.w_max = 0.95  # 最大惯性权重
        self.w_min = 0.4  # 最小惯性权重
        self.c1 = 1.5 # 认知系数
        self.c2 = 2 # 社会系数

        # 动态权重参数 (根据论文设置)
        self.beta = 10  # sigmoid陡度因子
        self.tau = 0.1 # 多样性中点
        self.epsilon = 1e-6  # 避免除零的小常数

        # 初始化粒子位置和速度
        self.X = np.random.uniform(low=[b[0] for b in bounds],
                                   high=[b[1] for b in bounds],
                                   size=(num_particles, dim))
        self.V = np.random.uniform(low=-1, high=1, size=(num_particles, dim))

        # 初始化个体最佳位置和全局最佳位置
        self.P = self.X.copy()
        self.pbest = np.array([cost_func(p) for p in self.P])
        self.gbest = self.P[np.argmin(self.pbest)]
        self.gbest_cost = min(self.pbest)

        self.P_std = self.X.copy()
        self.pbest_std = np.array([cost_func(p) for p in self.P_std])
        self.gbest_std = self.P_std[np.argmin(self.pbest_std)]
        self.gbest_cost_std = min(self.pbest_std)

        # 记录最佳成本历史和多样性历史
        self.cost_history = []
        self.cost_history_std = []
        self.diversity_history = []
        self.diversity_history_std = []
        self.w_history = []
        self.w_history_std = []

        # 计算最大可能方差(所有粒子在边界对角线上)
        self.sigma_max_sq = self._calculate_max_variance()

    def _calculate_max_variance(self):
        """计算搜索空间的最大可能方差"""
        # 假设粒子均匀分布在边界对角线上
        diagonal_length = np.sqrt(sum((b[1] - b[0]) ** 2 for b in self.bounds))
        return (diagonal_length / 2) ** 2

    def _calculate_diversity(self):
        """计算当前种群的多样性(方差)"""
        mean_position = np.mean(self.X, axis=0)
        distances_sq = np.sum((self.X - mean_position) ** 2, axis=1)
        return np.mean(distances_sq)

    def _update_inertia_weight(self, sigma_sq):
        """根据多样性更新惯性权重"""
        # 使用sigmoid函数动态调整权重
        normalized_sigma = sigma_sq / (self.sigma_max_sq + self.epsilon)
        sigmoid = 1 / (1 + np.exp(-self.beta * (normalized_sigma - self.tau)))
        return self.w_min + (self.w_max - self.w_min) * sigmoid

    def optimize(self):
        """执行优化"""
        # self.plot_2d_projection()
        for _ in range(self.max_iter):
            # 计算当前群体多样性
            current_diversity = self._calculate_diversity()
            self.diversity_history.append(current_diversity)

            # 更新惯性权重
            self.w = self._update_inertia_weight(current_diversity)
            self.w_history.append(self.w)

            # 更新速度和位置
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)

            self.V = (self.w * self.V +
                      self.c1 * r1 * (self.P - self.X) +
                      self.c2 * r2 * (self.gbest - self.X))

            self.X = self.X + self.V

            # 应用边界约束
            for i in range(self.dim):
                self.X[:, i] = np.clip(self.X[:, i], self.bounds[i][0], self.bounds[i][1])

            # 评估当前粒子位置
            costs = np.array([self.cost_func(p) for p in self.X])

            # 更新个体最佳
            improved_indices = costs < self.pbest
            self.P[improved_indices] = self.X[improved_indices]
            self.pbest[improved_indices] = costs[improved_indices]

            # 更新全局最佳
            if min(costs) < self.gbest_cost:
                self.gbest = self.X[np.argmin(costs)]
                self.gbest_cost = min(costs)

            # 记录历史最佳成本
            self.cost_history.append(self.gbest_cost)
        self.plot_2d_projection()
        # 初始化粒子位置和速度
        self.X = np.random.uniform(low=[b[0] for b in bounds],
                                   high=[b[1] for b in bounds],
                                   size=(num_particles, dim))
        self.V = np.random.uniform(low=-1, high=1, size=(num_particles, dim))
        return self.gbest, self.gbest_cost

    def optimize_std(self):
        self.c1 = 1.5
        self.c2 = 1.5
        self.plot_2d_projection()
        for _ in range(self.max_iter):
            # 计算当前群体多样性
            current_diversity = self._calculate_diversity()
            self.diversity_history_std.append(current_diversity)

            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)

            self.V = (self.w * self.V +
                      self.c1 * r1 * (self.P_std - self.X) +
                      self.c2 * r2 * (self.gbest_std - self.X))

            self.X = self.X + self.V

            self.w = self.w - 0.6 * (1 / self.max_iter)
            self.w_history_std.append(self.w)

            # 边界约束
            for i in range(self.dim):
                self.X[:, i] = np.clip(self.X[:, i], self.bounds[i][0], self.bounds[i][1])

            # 评估和更新
            costs = np.array([self.cost_func(p) for p in self.X])

            improved_indices = costs < self.pbest_std
            self.P_std[improved_indices] = self.X[improved_indices]
            self.pbest_std[improved_indices] = costs[improved_indices]

            if min(costs) < self.gbest_cost_std:
                self.gbest_std = self.X[np.argmin(costs)]
                self.gbest_cost_std = min(costs)

            self.cost_history_std.append(self.gbest_cost)
        self.plot_2d_projection()
        return self.gbest_std, self.gbest_cost_std

    def plot_convergence(self):
        """绘制收敛曲线和多样性变化"""
        plt.figure(figsize=(15, 10))

        # # 成本历史
        # plt.subplot(3, 1, 1)
        # plt.plot(self.cost_history)
        # plt.title('Convergence Curve')
        # plt.xlabel('Iteration')
        # plt.ylabel('Best Cost')
        # plt.grid(True)

        # 多样性历史
        plt.subplot(1, 2, 1)
        plt.plot(self.diversity_history, color='tab:blue',linewidth=2)
        plt.plot(self.diversity_history_std, color='tab:orange',linewidth=2)
        plt.title('Population Diversity')
        plt.xlabel('Iteration')
        plt.ylabel('Diversity (σ²)')
        plt.grid(True)

        # 惯性权重历史
        plt.subplot(1, 2, 2)
        plt.plot(self.w_history, color='tab:blue')
        # plt.plot(self.w_history_std, color='tab:orange')
        plt.title('Dynamic Inertia Weight')
        plt.xlabel('Iteration')
        plt.ylabel('Inertia Weight (w)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_2d_projection(self, X=None):
        """
        绘制粒子群的2D降维投影（简洁版）
        参数:
            X: 粒子位置数组 (None则使用self.X)
        """
        if X is None:
            X = self.X
        # 使用PCA降维到2D
        pca = PCA(n_components=2)
        projected = pca.fit_transform(X)

        # 创建图形
        plt.figure(figsize=(8, 8))
        plt.scatter(projected[:, 0], projected[:, 1],
                    c='orange', alpha=1, edgecolors='w', s=120)

        # 标记全局最优位置
        if hasattr(self, 'gbest'):
            gbest_projected = pca.transform(self.gbest.reshape(1, -1))
            plt.scatter(gbest_projected[:, 0], gbest_projected[:, 1],
                        marker='*', s=150, c='red', label='Global Best')

        plt.title('Projection of Particle Swarm Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

# 测试改进的PSO算法
if __name__ == "__main__":
    # 参数设置
    dim = 30  # 测试更高维度的问题
    num_particles = 50
    max_iter = 50
    bounds = [(-5.12, 5.12) for _ in range(dim)]  # Rastrigin函数的典型搜索空间


    # 创建并运行改进的PSO
    print("Running Diversity-Aware PSO...")
    dpso = DiversityAwarePSO(rastrigin, dim, num_particles, max_iter, bounds)
    best_solution, best_cost = dpso.optimize()
    best_solution_std, best_cost_std = dpso.optimize_std()

    # # 打印结果
    # print(f"\nBest solution found: {best_solution}")
    # print(f"Best cost: {best_cost}")

    # 绘制收敛曲线、多样性和权重变化
    dpso.plot_convergence()
