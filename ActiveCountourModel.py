import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.segmentation import active_contour

# 1. 读取图像并预处理
image = color.rgb2gray(io.imread('testmap.jpg'))
image = filters.gaussian(image, 1.0)

# 2. 初始化轮廓（例如圆形）
s = np.linspace(0, 2*np.pi, 400)
x = 150 + 100*np.cos(s)
y = 150 + 100*np.sin(s)
init = np.array([x, y]).T

# 3. 自定义收敛过程的迭代实现（保存中间轮廓）
from skimage.segmentation._active_contour import active_contour as internal_snake

snakes = []
snake = init.copy()
for i in range(100):  # 迭代100步
    snake = internal_snake(image, snake, alpha=0.015, beta=10, gamma=0.001, max_iterations=1)
    if i % 10 == 0:  # 每10步保存一次
        snakes.append(snake.copy())

# 4. 可视化多个中间轮廓
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
colors = plt.cm.viridis(np.linspace(0, 1, len(snakes)))

for i, s in enumerate(snakes):
    ax.plot(s[:, 0], s[:, 1], color=colors[i], label=f'Step {i*10}')
ax.legend()
plt.title('Active Contour Convergence')
plt.show()
