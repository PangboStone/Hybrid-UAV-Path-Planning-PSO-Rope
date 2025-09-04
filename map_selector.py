import rasterio
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Entry, Button, Label
from PIL import Image, ImageTk

class MapSelector:
    def __init__(self, tif_path):
        self.tif_path = tif_path
        self.start_pos = None
        self.target_pos = None
        self.fig, self.ax = plt.subplots()
        self.cid_move = None
        self.start_x_entry = None
        self.start_y_entry = None
        self.start_h_entry = None
        self.target_x_entry = None
        self.target_y_entry = None
        self.target_h_entry = None
        self.root = None

    def _on_mouse_move(self, event):
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= y < self.img.shape[0] and 0 <= x < self.img.shape[1]:
                z = self.img[y, x]
                self.ax.set_title(f'Current Position: X={x}, Y={y}, Z={z:.2f}')
                self.fig.canvas.draw_idle()

    def _confirm_positions(self):
        try:
            start_x = int(self.start_x_entry.get())
            start_y = int(self.start_y_entry.get())
            start_h = int(self.start_h_entry.get()) if self.start_h_entry.get() else 5
            with rasterio.open(self.tif_path) as src:
                start_z = src.read(1)[start_y, start_x] + start_h
            self.start_pos = (start_x, start_y, start_z)
        except ValueError:
            self.start_pos = None
            print("Invalid input for start position.")

        try:
            target_x = int(self.target_x_entry.get())
            target_y = int(self.target_y_entry.get())
            target_h = int(self.target_h_entry.get()) if self.target_h_entry.get() else 5
            with rasterio.open(self.tif_path) as src:
                target_z = src.read(1)[target_y, target_x] + target_h
            self.target_pos = (target_x, target_y, target_z)
        except ValueError:
            self.target_pos = None
            print("Invalid input for target position.")

        self.root.destroy()
        plt.close(self.fig)

    def select_points(self):
        with rasterio.open(self.tif_path) as src:
            self.img = src.read(1)
            plt.imshow(self.img, cmap='grey')

        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        self.root = tk.Tk()
        self.root.title("Select Positions")

        # Start Position Input
        start_frame = tk.Frame(self.root)
        start_frame.pack(pady=5)
        Label(start_frame, text="Input Start Position:  X:").pack(side=tk.LEFT)
        self.start_x_entry = Entry(start_frame, width=5)
        self.start_x_entry.pack(side=tk.LEFT)
        Label(start_frame, text=", Y:").pack(side=tk.LEFT)
        self.start_y_entry = Entry(start_frame, width=5)
        self.start_y_entry.pack(side=tk.LEFT)
        Label(start_frame, text=", h:").pack(side=tk.LEFT)
        self.start_h_entry = Entry(start_frame, width=5)
        self.start_h_entry.pack(side=tk.LEFT)

        # Target Position Input
        target_frame = tk.Frame(self.root)
        target_frame.pack(pady=5)
        Label(target_frame, text="Input Target Position: X:").pack(side=tk.LEFT)
        self.target_x_entry = Entry(target_frame, width=5)
        self.target_x_entry.pack(side=tk.LEFT)
        Label(target_frame, text=", Y:").pack(side=tk.LEFT)
        self.target_y_entry = Entry(target_frame, width=5)
        self.target_y_entry.pack(side=tk.LEFT)
        Label(target_frame, text=", h:").pack(side=tk.LEFT)
        self.target_h_entry = Entry(target_frame, width=5)
        self.target_h_entry.pack(side=tk.LEFT)

        # Confirm Button
        confirm_button = Button(self.root, text="Confirm", command=self._confirm_positions)
        confirm_button.pack(pady=10)

        plt.show(block=False) # Show the matplotlib plot
        self.root.mainloop() # Start the tkinter event loop

        return self.start_pos, self.target_pos

if __name__ == '__main__':
    # 示例用法：请替换为你的 TIF 文件路径
    tif_file_path = 'heightmap.tiff'
    selector = MapSelector(tif_file_path)
    start, target = selector.select_points()
    if start and target:
        print(f"Start Position: {start}")
        print(f"Target Position: {target}")
    else:
        print("Failed to get valid start and target positions.")