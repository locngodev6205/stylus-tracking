import tkinter as tk

import cv2
import numpy as np
from PIL import ImageTk, Image

from stylus_tracking.capture.video_capture import VideoCapture
from stylus_tracking.controller.controller import Controller
from stylus_tracking.ui.graph import Graph


class App:
    DELAY = 1
    RESIZE_FACTOR = 1
    COLOR = "#e6e6e6"

    def __init__(self, window: tk.Tk, window_title: str, controller: Controller, logger):
        self.window = window

        self.window.title(window_title)

        self.window.bind('<Escape>', lambda e: window.quit())
        self.window.config(background=self.COLOR)

        self.controller = controller

        self.current_graph = Graph(self.window, 10, 8)
        self.canvas_drawing = tk.Canvas(self.window,
                                         width=self.controller.model.drawing.shape[1],
                                         height=self.controller.model.drawing.shape[0],
                                         background=self.COLOR)
        self.canvas_drawing.grid(row=1, column=1)

        self.logger = logger

        self.camera_frame = None
        self.camera_canvas = None

        self.reset_graph = tk.Button(self.window,
                                     text="Reset graph",
                                     command=self.__reset_graph)
        self.reset_graph.grid(row=2, column=1, pady=5)

        self.calibration_button = tk.Button(self.window,
                                             text="Calibration window",
                                             command=self.__calibration_child)
        self.calibration_button.grid(row=3, column=1, pady=5)

        # Khung hiển thị tọa độ đầu bút
        self.coord_frame = tk.LabelFrame(self.window, text=" Tọa độ đầu bút (Stylus Tip Position) ",
                                         font=("Helvetica", 11, "bold"), fg="#333333",
                                         background=self.COLOR, bd=2, relief="groove")
        self.coord_frame.grid(row=4, column=1, pady=10, padx=10, sticky="ew")

        self.coord_x = tk.Label(self.coord_frame, text="X: --- mm", font=("Consolas", 12, "bold"), fg="#777777", background=self.COLOR)
        self.coord_x.grid(row=1, column=1, padx=15, pady=5)
        
        self.coord_y = tk.Label(self.coord_frame, text="Y: --- mm", font=("Consolas", 12, "bold"), fg="#777777", background=self.COLOR)
        self.coord_y.grid(row=1, column=2, padx=15, pady=5)
        
        self.coord_z = tk.Label(self.coord_frame, text="Z: --- mm", font=("Consolas", 12, "bold"), fg="#777777", background=self.COLOR)
        self.coord_z.grid(row=1, column=3, padx=15, pady=5)

        self.current_image = None

        self.__update()

        self.window.mainloop()

    def __update(self):
        refresh = self.controller.next_frame()
        if self.camera_frame is not None:
            resized_image = cv2.resize(self.controller.model.current_frame, None,
                                       fx=self.RESIZE_FACTOR, fy=self.RESIZE_FACTOR, interpolation=cv2.INTER_LINEAR)
            self.current_image = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
            self.camera_canvas.create_image(0, 0, image=self.current_image, anchor=tk.NW)

        self.current_drawing = ImageTk.PhotoImage(image=Image.fromarray(self.controller.model.drawing))
        self.canvas_drawing.create_image(0, 0, image=self.current_drawing, anchor=tk.NW)

        # Cập nhật tọa độ lên giao diện GUI
        new_x = self.controller.model.new_x
        new_y = self.controller.model.new_y
        new_z = self.controller.model.new_z
        if new_x is not None and new_y is not None and new_z is not None:
            self.coord_x.config(text=f"X: {new_x:7.1f} mm", fg="#d32f2f")
            self.coord_y.config(text=f"Y: {new_y:7.1f} mm", fg="#388e3c")
            # Màu cho Z: xanh lá nếu sát mặt bàn (|z| < 20), vàng nếu hơi cao, đỏ nếu quá cao
            if abs(new_z) < 20:
                z_color = "#388e3c" # xanh lá
            elif abs(new_z) < 80:
                z_color = "#f57c00" # vàng/cam
            else:
                z_color = "#d32f2f" # đỏ
            self.coord_z.config(text=f"Z: {new_z:7.1f} mm", fg=z_color)
        else:
            self.coord_x.config(text="X: --- mm", fg="#777777")
            self.coord_y.config(text="Y: --- mm", fg="#777777")
            self.coord_z.config(text="Z: --- mm", fg="#777777")

        # self.__update_graphic()
        self.window.after(self.DELAY, self.__update)

    def __update_graphic(self):
        if self.controller.model.new_x is not None:
            self.current_graph.update(self.controller.model.new_x, self.controller.model.new_y,
                                      self.controller.model.new_z)

    def __reset_graph(self):
        # self.current_graph.reset()
        self.controller.model.reset_graph()

    def __calibration_child(self):
        self.camera_frame = tk.Toplevel(self.window)
        self.camera_frame.bind('<Escape>', self.__close_camera_frame)
        self.camera_frame.protocol('WM_DELETE_WINDOW', self.__close_camera_frame)
        self.camera_canvas = tk.Canvas(self.camera_frame,
                                       width=VideoCapture.WIDTH * self.RESIZE_FACTOR,
                                       height=VideoCapture.HEIGHT * self.RESIZE_FACTOR,
                                       background=self.COLOR)

        self.camera_canvas.grid(row=2, column=1)
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
