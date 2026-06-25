import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class Graph:
    def __init__(self, master, width, height):
        plt.xkcd() # vẽ theo phong các truyện tranh
        fig = Figure(figsize=(width, height))
        
        # vẽ không gian 3D
        self.ax = Axes3D(fig)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        self.ax.mouse_init()

        self.graph_canvas = FigureCanvasTkAgg(fig, master=master) # gắn lên cửa sổ master
        self.graph_canvas.get_tk_widget().grid(row=1, column=1) # sắp xếp biểu đồ

    def update(self, x, y, z):
        self.ax.scatter(x, y, z)
        self.graph_canvas.draw()

    def reset(self):
        self.ax.clear()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
