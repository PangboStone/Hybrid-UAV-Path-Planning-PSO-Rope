import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle, Rectangle


class InteractiveRopeModel:
    def __init__(self, start_point, end_point, t, d, k, grid_size=500, c_thr = 0.001 ):
        """
        初始化增强版绳模型

        参数:
        - start_point: (a1, b1) 起点坐标
        - end_point: (a2, b2) 终点坐标
        - k: 中间节点数量
        - grid_size: 网格大小 (n*n)
        """
        self.start = np.array(start_point, dtype=float)
        self.end = np.array(end_point, dtype=float)
        self.k = k
        self.grid_size = grid_size
        self.nodes = self.initialize_nodes()

        # 物理参数
        self.tension = t  # 张力系数
        self.damping = d  # 阻尼系数
        self.max_iterations = 1000  # 最大迭代次数
        self.convergence_threshold = c_thr  # 收敛阈值

        # 交互状态
        self.dragging = False
        self.drag_node = None
        self.length_history = []

        # 设置可视化
        self.setup_visualization()

    def initialize_nodes(self):
        """初始化节点，在网格中随机生成节点位置"""
        nodes = [self.start.copy()]  # 添加起点

        # 随机生成k个中间节点
        for _ in range(self.k):
            # 在网格范围内随机生成坐标
            x = np.random.uniform(0, self.grid_size)
            y = np.random.uniform(0, self.grid_size)
            nodes.append(np.array([x, y]))

        nodes.append(self.end.copy())  # 添加终点
        return np.array(nodes)

    def apply_constraints(self):
        """应用约束条件"""
        # 固定起点和终点
        if not self.dragging or self.drag_node != 0:
            self.nodes[0] = self.start.copy()
        if not self.dragging or self.drag_node != len(self.nodes) - 1:
            self.nodes[-1] = self.end.copy()


    def update_nodes(self):
        """更新节点位置"""
        new_nodes = self.nodes.copy()

        for i in range(1, len(self.nodes) - 1):
            # 计算来自左右节点的拉力
            left_vec = self.nodes[i - 1] - self.nodes[i]
            right_vec = self.nodes[i + 1] - self.nodes[i]

            # 计算合力
            force = self.tension * (left_vec + right_vec)

            # 更新位置 (简单的欧拉积分)
            new_nodes[i] = self.nodes[i] + force - self.damping * force

        self.nodes = new_nodes
        self.apply_constraints()


    def calculate_total_length(self):
        """计算绳子总长度"""
        total_length = 0
        for i in range(len(self.nodes) - 1):
            total_length += np.linalg.norm(self.nodes[i + 1] - self.nodes[i])
        return total_length

    def on_click(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.ax:
            return

        # 检查是否点击了起点或终点
        start_dist = np.linalg.norm(np.array([event.xdata, event.ydata]) - self.start)
        end_dist = np.linalg.norm(np.array([event.xdata, event.ydata]) - self.end)

        if start_dist < 20:  # 点击了起点
            self.dragging = True
            self.drag_node = 0
        elif end_dist < 20:  # 点击了终点
            self.dragging = True
            self.drag_node = len(self.nodes) - 1

    def on_release(self, event):
        """处理鼠标释放事件"""
        self.dragging = False
        self.drag_node = None

    def on_motion(self, event):
        """处理鼠标移动事件"""
        if not self.dragging or event.inaxes != self.ax:
            return

        # 更新被拖动的节点位置
        new_pos = np.array([event.xdata, event.ydata])

        if self.drag_node == 0:  # 拖动起点
            self.start = new_pos
        elif self.drag_node == len(self.nodes) - 1:  # 拖动终点
            self.end = new_pos

        self.apply_constraints()
        self.update_plot(0)

    def reset_simulation(self, event):
        """重置模拟"""
        self.nodes = self.initialize_nodes()
        self.length_history = []
        self.update_plot(0)

    def run_simulation(self, event):
        """运行模拟"""
        self.length_history = []  # 清空历史记录

        for iteration in range(self.max_iterations):
            self.update_nodes()
            current_length = self.calculate_total_length()
            self.length_history.append(current_length)

            # 更新长度曲线
            self.length_line.set_data(range(len(self.length_history)), self.length_history)
            self.length_ax.relim()
            self.length_ax.autoscale_view()

            # 更新主图
            self.update_plot(iteration)

            # 检查收敛
            if iteration > 10 and abs(self.length_history[-1] - self.length_history[-2]) < self.convergence_threshold:
                print(f"Converge at {iteration}")
                break

            plt.pause(0.01)

        print("Simulation Finished")

    def setup_visualization(self):
        """设置可视化界面"""
        plt.close('all')
        self.fig = plt.figure(figsize=(12, 6))

        # 主图 - 绳模型
        self.ax = self.fig.add_subplot(121)
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('Rope Model Simulation')


        # 绘制起点和终点
        self.start_point, = self.ax.plot(*self.start, 'go', markersize=10, label='START')
        self.end_point, = self.ax.plot(*self.end, 'ro', markersize=10, label='END')

        # 初始化绳子绘制
        self.line, = self.ax.plot(self.nodes[:, 0], self.nodes[:, 1], 'b-o', linewidth=2, markersize=5, label='Rope Node')
        self.ax.legend()

        # 长度曲线图
        self.length_ax = self.fig.add_subplot(122)
        self.length_ax.set_xlabel('Iteration')
        self.length_ax.set_ylabel('Rope Length')
        self.length_ax.set_title('Length Change Curve')
        self.length_line, = self.length_ax.plot([], [], 'b-', lw=3)

        # 添加按钮
        self.button_ax = plt.axes([0.3, -0.02, 0.15, 0.05])
        self.run_button = Button(self.button_ax, 'Activate')
        self.run_button.on_clicked(self.run_simulation)

        self.reset_ax = plt.axes([0.5, -0.02, 0.15, 0.05])
        self.reset_button = Button(self.reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset_simulation)

        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        plt.tight_layout()

    def update_plot(self, iteration):
        """更新绘图"""
        # 更新绳子
        self.line.set_data(self.nodes[:, 0], self.nodes[:, 1])

        # 更新起点和终点 - 修正这里
        self.start_point.set_data([self.start[0]], [self.start[1]])
        self.end_point.set_data([self.end[0]], [self.end[1]])

        # 更新标题
        self.ax.set_title(f'Iteration: {iteration}, Rope Length: {self.calculate_total_length():.2f}')

        self.fig.canvas.draw_idle()


# 运行模拟
if __name__ == "__main__":
    # 参数设置
    start = (50, 50)
    end = (450, 450)
    k = 30  # 中间节点数
    grid_size = 500
    t = 1.6  # 张力系数
    d = 0.7  # 阻尼系数
    c_thr = 0.05

    # 创建并显示模型
    model = InteractiveRopeModel(start, end, t, d, k, grid_size, c_thr)
    plt.show()