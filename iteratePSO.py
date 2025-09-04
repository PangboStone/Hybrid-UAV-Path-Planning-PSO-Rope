from update import update_velocity, update_position
from calFitness import cost_function
import numpy as np


# def Iterate(self):
#     """ 进行迭代 """
#     for it in range(self.MaxIt):
#         for particle in self.particles:
#             particle['velocity'] = update_velocity(self, particle)
#             particle['position'] = np.round(update_position(particle))
#             path_length, _, cost = cost_function(self, particle['position'])
#             particle['cost'] = cost
#
#
#             # 求解个体最优 ‘best_position’
#             if cost < particle['best_cost']:
#                 particle['best_position'] = particle['position'].copy()
#                 particle['best_cost'] = cost
#                 # 求解出个体最优后继续求解全体最优
#                 if cost < self.global_best['cost']:
#                     self.global_best['position'] = particle['position'].copy()
#                     self.global_best['cost'] = cost
#                     print(f"新的全局最优: 位置 = {self.global_best['position']}, 成本 = {self.global_best['cost']}, 迭代次数{it}")
#         # print(f"当前状态: 位置 = {particle['best_position']}, 成本 = {particle['best_cost']}, 迭代次数{it}")
#         self.w *= 0.98  # 减小惯性权重
#    return self.global_best['position'], self.global_best['cost']

"2.0 迭代机制，具有早停功能"

#
# def Iterate(self):
#     """ 进行迭代 """
#     best_cost_history = []
#     no_improvement_counter = 0
#     early_stopping_threshold = 200
#     min_cost_change = 1e-3  # 定义小于此变化量为没有显著变化
#
#     for it in range(self.MaxIt):
#         previous_global_best_cost = self.global_best['cost']
#
#         for particle in self.particles:
#             particle['velocity'] = update_velocity(self, particle)
#             particle['position'] = np.round(update_position(particle))
#             path_length, _, cost = cost_function(self, particle['position'])
#             particle['cost'] = cost
#
#             # 求解个体最优 ‘best_position’
#             if cost < particle['best_cost']:
#                 particle['best_position'] = particle['position'].copy()
#                 particle['best_cost'] = cost
#                 # 求解出个体最优后继续求解全体最优
#                 if cost < self.global_best['cost']:
#                     self.global_best['position'] = particle['position'].copy()
#                     self.global_best['cost'] = cost
#                     print(f"新的全局最优: 位置 = {self.global_best['position']}, 成本 = {self.global_best['cost']}, 迭代次数{it}")
#
#         # 检查全局最优是否发生显著变化
#         cost_change = abs(previous_global_best_cost - self.global_best['cost'])
#         if cost_change < min_cost_change:
#             no_improvement_counter += 1
#         else:
#             no_improvement_counter = 0
#
#         # 应用早停机制
#         if no_improvement_counter >= early_stopping_threshold:
#             print(f"早停触发: 全局最优在最近 {early_stopping_threshold} 次迭代中变化小于 {min_cost_change}。")
#             break
#
#         self.w *= 0.98  # 减小惯性权重
#
#     print(f"最终全局最优: 位置 = {self.global_best['position']}, 成本 = {self.global_best['cost']}")


"3.0 迭代机制，具有早停和动态调整参数机制"
import numpy as np
import matplotlib.pyplot as plt

def Iterate(self):
    """ 进行迭代，并记录全局最优成本 """
    best_cost_history = []
    no_improvement_counter = 0
    early_stopping_threshold = 10
    min_cost_change = 1e-2  # 定义小于此变化量为没有显著变化
    early_stopped = False
    restart_iterations = 100  # 早停后重新搜索的迭代次数
    initial_w = self.w  # 保存初始的惯性权重

    for it in range(self.MaxIt):
        previous_global_best_cost = self.global_best['cost']
        best_cost_history.append(self.global_best['cost']) # 记录当前全局最优成本

        for particle in self.particles:
            particle['velocity'] = update_velocity(self, particle)
            particle['position'] = np.round(update_position(particle))
            path_length, _, cost = cost_function(self, particle['position'])
            particle['cost'] = cost

            # 求解个体最优 ‘best_position’
            if cost < particle['best_cost']:
                particle['best_position'] = particle['position'].copy()
                particle['best_cost'] = cost
                # 求解出个体最优后继续求解全体最优
                if cost < self.global_best['cost']:
                    self.global_best['position'] = particle['position'].copy()
                    self.global_best['cost'] = cost
                    print(f"新的全局最优: 位置 = {self.global_best['position']}, 成本 = {self.global_best['cost']}, 迭代次数{it}")

        # 检查全局最优是否发生显著变化
        cost_change = abs(previous_global_best_cost - self.global_best['cost'])
        if cost_change < min_cost_change and not early_stopped:
            no_improvement_counter += 1
        else:
            no_improvement_counter = 0

        # 应用早停机制
        if no_improvement_counter >= early_stopping_threshold and not early_stopped:
            print(f"早停触发: 全局最优在最近 {early_stopping_threshold} 次迭代中变化小于 {min_cost_change}。")
            self.w = 0.9  # 增大惯性权重，鼓励全局搜索
            # 可以选择性地增大 c2
            self.c2 = min(self.c2 + 0.5, 3)
            early_stopped = True
            restart_counter = 0 # 记录重新搜索的迭代次数

        # 早停后进行重新搜索的阶段
        if early_stopped:
            restart_counter += 1
            # 在重新搜索的后期逐渐恢复惯性权重
            self.w = max(0.4, 0.9 - 0.5 * (restart_counter / restart_iterations))
            # 也可以逐渐恢复 c2

            if restart_counter >= restart_iterations:
                print("重新搜索阶段结束，恢复正常收敛。")
                early_stopped = False

        elif not early_stopped:
            self.w *= 0.98  # 正常减小惯性权重
        if early_stopped:
            print(f"重新搜索迭代次数: {it}, w = {self.w:.3f}, 全局最优成本 = {self.global_best['cost']:.6f}")
        # print(f"当前状态: 迭代次数{it}, w = {self.w:.3f}, 全局最优成本 = {self.global_best['cost']:.6f}")

        if it > self.MaxIt: # 确保不会无限循环
            break

    print(f"最终全局最优: 位置 = {self.global_best['position']}, 成本 = {self.global_best['cost']}")

    # 绘制成本历史曲线
    plt.figure()
    plt.plot(range(len(best_cost_history)), best_cost_history, lw = 3)
    plt.xlabel('Iteration')
    plt.ylabel('Global Best Cost')
    plt.title('Global Best Cost over Iterations')
    plt.grid(True)
    plt.show()