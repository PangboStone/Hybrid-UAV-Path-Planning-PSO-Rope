import rasterio
import matplotlib.pyplot as plt
import numpy as np

class RasterMapExplorer:
    def __init__(self, tiff_file):
        """
        初始化栅格地图浏览器。

        Args:
            tiff_file (str): TIFF栅格地图文件的路径。
        """
        self.tiff_file = tiff_file
        self.dataset = None
        self.image = None
        self.extent = None
        self.fig, self.ax = plt.subplots()
        self.connect()

    def load_raster(self):
        """
        加载栅格数据。
        """
        try:
            self.dataset = rasterio.open(self.tiff_file)
            self.image = self.dataset.read(1)  # 假设只有一个波段代表高度
            # 计算图像的地理范围 (left, right, bottom, top)
            left, bottom, right, top = self.dataset.bounds
            self.extent = (left, right, bottom, top)
        except rasterio.RasterioIOError:
            print(f"无法打开文件: {self.tiff_file}")
            self.image = None
            self.extent = None

    def visualize_map(self):
        """
        使用matplotlib可视化地图。
        """
        if self.image is not None:
            self.ax.imshow(self.image, cmap='grey', extent=self.extent)
            self.ax.set_title('Grid Map (Click to query)')
            self.fig.canvas.draw()
        else:
            print("请先加载栅格数据。")

    def get_height(self, x, y):
        """
        获取指定地理坐标的高度值。

        Args:
            x (float): X坐标。
            y (float): Y坐标。

        Returns:
            float or None: 该坐标的高度值，如果超出范围则返回None。
        """
        if self.dataset is not None:
            try:
                row, col = self.dataset.index(x, y)
                if 0 <= row < self.image.shape[0] and 0 <= col < self.image.shape[1]:
                    return self.image[row, col]
                else:
                    return None
            except rasterio.transform.OutsideBounds:
                return None
        else:
            return None

    def onclick(self, event):
        """
        处理鼠标点击事件，查询并显示点击位置的高度。
        """
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                height = self.get_height(x, y)
                if height is not None:
                    print(f"点击位置 (x={x:.2f}, y={y:.2f}) 的高度为: {height}")
                    self.ax.set_title(f'Coordinate (x={x:.2f}, y={y:.2f}) --- Height: {height}')
                    self.fig.canvas.draw()
                else:
                    print(f"点击位置 (x={x:.2f}, y={y:.2f}) 超出地图范围。")

    def connect(self):
        """
        连接鼠标点击和文本输入事件。
        """
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)


    def show(self):
        """
        显示地图和交互界面。
        """
        self.load_raster()
        if self.image is not None:
            self.visualize_map()
            plt.show()

if __name__ == '__main__':
    # 将 'your_tiff_file.tif' 替换为你的实际文件路径
    tiff_file_path = 'testmap.tiff'
    explorer = RasterMapExplorer(tiff_file_path)
    explorer.show()