import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class Graph:
    def __init__(self, master, width, height):
        # plt.xkcd() # Đã tắt: chế độ vẽ truyện tranh này gây lỗi thiếu font trên Linux
        fig = Figure(figsize=(width, height))
        
        # vẽ không gian 3D
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        # Khóa tỷ lệ các trục (tùy chọn để hình khỏi móp)
        # self.ax.set_box_aspect([1, 1, 1])

        # Vẽ 3 trục tọa độ gốc (0,0,0) dài 150mm để định hướng
        AXIS_LENGTH = 150
        # Trục X (Đỏ) - Thường là sang ngang
        self.ax.plot([0, AXIS_LENGTH], [0, 0], [0, 0], color='red', linewidth=3)
        self.ax.text(AXIS_LENGTH + 10, 0, 0, 'X', color='red', weight='bold')
        
        # Trục Y (Xanh lá) - Thường là dọc theo bàn
        self.ax.plot([0, 0], [0, AXIS_LENGTH], [0, 0], color='green', linewidth=3)
        self.ax.text(0, AXIS_LENGTH + 10, 0, 'Y', color='green', weight='bold')
        
        # Trục Z (Xanh dương) - Thẳng đứng lên trời
        self.ax.plot([0, 0], [0, 0], [0, AXIS_LENGTH], color='blue', linewidth=3)
        self.ax.text(0, 0, AXIS_LENGTH + 10, 'Z', color='blue', weight='bold')

        self.ax.mouse_init()

        self.graph_canvas = FigureCanvasTkAgg(fig, master=master) # gắn lên cửa sổ master
        self.graph_canvas.get_tk_widget().grid(row=1, column=2, padx=10) # Đưa 3D sang cột 2 để không đè lên 2D

        # Mảng lưu lịch sử tọa độ
        self.x_data = []
        self.y_data = []
        self.z_data = []
        # Khởi tạo sẵn 1 đường vẽ 3D liên tục (nhanh hơn scatter rất nhiều)
        self.line, = self.ax.plot([], [], [], color='#d32f2f', linewidth=2, marker='o', markersize=2)

        # Ép canvas render đồ thị 3D (và 3 trục tọa độ) lên màn hình ngay khi khởi động
        self.graph_canvas.draw()

    def update(self, x, y, z):
        # Lưu điểm mới vào mảng
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)
        
        # Cập nhật dữ liệu cho đường line có sẵn thay vì vẽ mới
        self.line.set_data(self.x_data, self.y_data)
        self.line.set_3d_properties(self.z_data)
        
        # Tự động scale camera để luôn nhìn thấy toàn bộ nét vẽ
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Dùng draw_idle() để tối ưu vòng lặp GUI (không block màn hình)
        self.graph_canvas.draw_idle()

    def reset(self):
        # Xóa dữ liệu mảng
        self.x_data.clear()
        self.y_data.clear()
        self.z_data.clear()
        
        # Cập nhật lại đường vẽ rỗng
        self.line.set_data(self.x_data, self.y_data)
        self.line.set_3d_properties(self.z_data)
        self.graph_canvas.draw_idle()
