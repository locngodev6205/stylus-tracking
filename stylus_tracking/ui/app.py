import tkinter as tk

import cv2
import numpy as np
from PIL import ImageTk, Image

from stylus_tracking.capture.video_capture import VideoCapture
from stylus_tracking.controller.controller import Controller
from stylus_tracking.ui.graph import Graph


class App:
    DELAY = 1 # 1ms
    RESIZE_FACTOR = 1
    COLOR = "#e6e6e6"

    def __init__(self, window: tk.Tk, window_title: str, controller: Controller, logger):
        # window co giản tùy element trên nó
        self.window = window
        
        # cấu hình window
        self.window.title(window_title)
        self.window.bind('<Escape>', lambda e: window.quit())
        self.window.config(background=self.COLOR)

        self.controller = controller

        # vẽ 3D
        self.current_graph = Graph(self.window, 10, 8) # không dùng đến

        # khung draw ở win1
        self.canvas_drawing = tk.Canvas(self.window,
                                         width=self.controller.model.drawing.shape[1],
                                         height=self.controller.model.drawing.shape[0],
                                         background=self.COLOR)
        self.canvas_drawing.grid(row=1, column=1)

        self.logger = logger

        # window chứa element thứ 2
        self.camera_frame = None
        # khụng để vẽ hình ảnh camera ở win2
        self.camera_canvas = None

        # vẽ button
        self.reset_graph = tk.Button(self.window,
                                     text="Reset graph",
                                     command=self.__reset_graph)
        self.reset_graph.grid(row=2, column=1, pady=5)

        self.calibration_button = tk.Button(self.window,
                                             text="Calibration window",
                                             command=self.__calibration_child)
        self.calibration_button.grid(row=3, column=1, pady=5)

        self.current_image = None

        self.__update()

        # duy trì cửa sổ
        self.window.mainloop()


    # hàm lặp chính
    def __update(self):
        # trả về màn hình có được vẽ mới không
        refresh = self.controller.next_frame()

        # kiểm tra màn hình thứ 2 được bật chưa
        if self.camera_frame is not None:
            # thi phóng hình ảnh để lên window thứ 2
            resized_image = cv2.resize(self.controller.model.current_frame, None,
                                       fx=self.RESIZE_FACTOR, fy=self.RESIZE_FACTOR, interpolation=cv2.INTER_LINEAR)
            
            # # --- VẼ TỌA ĐỘ TRỰC TIẾP LÊN HÌNH ẢNH CAMERA ---
            # new_x = self.controller.model.new_x
            # new_y = self.controller.model.new_y
            # new_z = self.controller.model.new_z
            
            # if new_x is not None and new_y is not None and new_z is not None:
            #     text = f"X: {new_x:5.1f}  Y: {new_y:5.1f}  Z: {new_z:5.1f} (mm)"
            #     # Tạo viền đen cho chữ dễ đọc trên nền sáng
            #     cv2.putText(resized_image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            #     # Chữ chính màu xanh lá
            #     cv2.putText(resized_image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            # else:
            #     text = "Tracking Lost / No Stylus Detected"
            #     cv2.putText(resized_image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            #     cv2.putText(resized_image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            # chuyển về đúng định dạng mà lib Tkinter dùng
            # hình ảnh của camera (win2)
            self.current_image = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
            self.camera_canvas.create_image(0, 0, image=self.current_image, anchor=tk.NW)

        # hình ảnh vẽ (win1)
        self.current_drawing = ImageTk.PhotoImage(image=Image.fromarray(self.controller.model.drawing))
        self.canvas_drawing.create_image(0, 0, image=self.current_drawing, anchor=tk.NW)

        self.__update_graphic()
        # delay 1ms rồi mới gọi update
        self.window.after(self.DELAY, self.__update)

    def __update_graphic(self):
        if self.controller.model.new_x is not None:
            self.current_graph.update(self.controller.model.new_x, self.controller.model.new_y,
                                      self.controller.model.new_z)

    def __reset_graph(self):
        # self.current_graph.reset()
        self.controller.model.reset_graph()

    def __calibration_child(self):
        # tạo một window mới 
        self.camera_frame = tk.Toplevel(self.window)
        self.camera_frame.bind('<Escape>', self.__close_camera_frame)
        self.camera_frame.protocol('WM_DELETE_WINDOW', self.__close_camera_frame)

        # gắn nơi render hình ảnh từ camera lên camera frame
        self.camera_canvas = tk.Canvas(self.camera_frame,
                                       width=VideoCapture.WIDTH * self.RESIZE_FACTOR,
                                       height=VideoCapture.HEIGHT * self.RESIZE_FACTOR,
                                       background=self.COLOR)
        self.camera_canvas.grid(row=1, column=1)


        # button
        calibration_buttons = tk.Canvas(self.camera_frame,
                                        width=self.window.winfo_width(),
                                        background=self.COLOR)
        calibration_buttons.grid(row=3, column=1, columnspan=2)
        calibrate_intrinsic_button = tk.Button(calibration_buttons,
                                               text="Calibrate intrinsic",
                                               command=self.controller.start_intrinsic_calibration)
        calibrate_intrinsic_button.grid(row=1, column=1)
        load_previous_intrinsic_parameters_button = tk.Button(calibration_buttons,
                                                              text="Load previous intrinsic parameters",
                                                              command=self.controller.try_load_previous_intrinsic_calibration_parameters)
        load_previous_intrinsic_parameters_button.grid(row=1, column=2)
        calibrate_extrinsic_button = tk.Button(calibration_buttons,
                                               text="Calculate extrinsic from intrinsic parameters",
                                               command=self.controller.calculate_extrinsic)
        calibrate_extrinsic_button.grid(row=2, column=1, columnspan=2)

        done_button = tk.Button(calibration_buttons,
                                text="Done",
                                command=self.__close_camera_frame)
        done_button.grid(row=4, column=1, columnspan=2)

    def __close_camera_frame(self, event=None):
        if self.camera_frame is not None:
            self.camera_frame.destroy()
            self.camera_frame = None
