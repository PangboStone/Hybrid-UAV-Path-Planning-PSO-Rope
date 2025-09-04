import random
import numpy as np
from numpy import round


def update_velocity(self, particle):
    """ 更新速度 """
    new_velocity = []
    for i in range(len(particle['position'])):
        r1, r2 = random.random(), random.random()
        cognitive = self.c1 * r1 * (np.array(particle['best_position'][i]) - np.array(particle['position'][i]))
        social = self.c2 * r2 * (np.array(self.global_best['position'][i]) - np.array(particle['position'][i]))
        new_velocity.append(self.w * particle['velocity'][i] + cognitive + social)
    return round(new_velocity)

def update_position(particle):
    """ 更新位置 """
    new_position = []
    # for i in range(len(particle['position'])):
    #     new_x = particle['position'][i][0] + particle['velocity'][i][0]
    #     new_y = particle['position'][i][1] + particle['velocity'][i][1]
    #     new_z = particle['position'][i][2] + particle['velocity'][i][2]
    #     new_position.append((new_x, new_y, new_z))
    # return new_position
    for i in range(len(particle['position'])):
        # 更新每个三维点的位置
        new_position.append(particle['position'][i] + particle['velocity'][i])
    return np.array(round(new_position).astype(int))  # 返回新的位置作为 numpy 数组


