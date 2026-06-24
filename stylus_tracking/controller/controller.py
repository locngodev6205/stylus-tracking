import sys
# quản lý ghi log, thông báo lỗi/tiến trình
from logging import Logger
import numpy as np

from stylus_tracking.calibration import calibration
from stylus_tracking.calibration.calibration import State
from stylus_tracking.capture.video_capture import VideoCapture
from stylus_tracking.controller.model import AppModel
from stylus_tracking.detection import detection
from stylus_tracking.filter.filter import FilterNone, FilterMedian, FilterKalman


class Controller:

    BUFFER_SIZE = 9
    # FILTER_TYPE = "median"
    FILTER_TYPE = "kalman"

    # CHỌN CAMERA: "phone" (dùng DroidCam) hoặc "pc" (dùng Webcam máy tính)
    CAMERA_MODE = "phone" 

    # BẬT NẾU HÌNH ẢNH BỊ LẬT NGANG (mirror) - thường xảy ra khi DroidCam dùng camera trước
    FLIP_HORIZONTAL = True

    # ĐỔI IP THÀNH IP ĐIỆN THOẠI CỦA BẠN (xem trong app DroidCam)
    DROIDCAM_URL = "http://192.168.1.13:4747/video"

    def __init__(self, logger: Logger, video_source=None):
        self.logger = logger
        if video_source is None:
            if self.CAMERA_MODE == "phone":
                video_source = self.DROIDCAM_URL
            else:
                video_source = 0  # 0 là camera mặc định của máy tính
        self.video_capture = VideoCapture(video_source, flip_horizontal=self.FLIP_HORIZONTAL)
        self.calibration = calibration.Calibration(self.logger.getChild("Calibration"))
        self.state = State.RAW
        self.detection = None

        self.model = AppModel()

        if self.FILTER_TYPE == "median":
            self.filter = FilterMedian()
        elif self.FILTER_TYPE == "kalman":
            self.filter = FilterKalman()
        elif self.FILTER_TYPE == "none":
            self.filter = FilterNone()

    # GIAI ĐOẠN 4: NHẬN DẠNG VÀ BẮT TỌA ĐỘ BÚT REAL-TIME
    def next_frame(self):
        ret, frame = self.video_capture.get_next_frame()
        self.model.current_frame = frame
        refresh = False
        if ret:
            self.model.current_frame = frame
            if self.state is State.CALIBRATING_INTRINSIC:
                if self.calibration.calculate_intrinsic(frame):
                    self.state = State.CALIBRATED_INTRINSIC
            if self.state is State.CALIBRATING_EXTRINSIC:
                if self.calibration.calculate_extrinsic(self.model.current_frame):
                    self.state = State.CALIBRATED
                    self.detection = detection.Detection(self.calibration)
                else:
                    self.state = State.CALIBRATED_INTRINSIC
            if self.state is State.CALIBRATED:
                if self.detection is not None:
                    self.model.current_frame, point = self.detection.detect(frame)
                    refresh = self.filter_and_add_point(point)
                else:
                    self.logger.info("Calibration should be performed prior to detection.")
        return refresh

    # GIAI ĐOẠN 2: HIỆU CHUẨN NỘI THAM (INTRINSIC CALIBRATION)
    def start_intrinsic_calibration(self) -> None:
        self.state = State.CALIBRATING_INTRINSIC
        self.calibration.start_intrinsic_calibration()

    # GIAI ĐOẠN 3: HIỆU CHUẨN NGOẠI THAM (EXTRINSIC CALIBRATION)
    def calculate_extrinsic(self) -> None:
        if self.state is not State.CALIBRATED_INTRINSIC:
            self.logger.info("Intrinsic calibration should be performed prior to the extrinsic one.")
        else:
            self.state = State.CALIBRATING_EXTRINSIC

    # kiểm tra nếu đã có dữ liệu hiệu chuẩn nội tham trước đó thì load lên
    def try_load_previous_intrinsic_calibration_parameters(self) -> None:
        if self.calibration.try_load_intrinsic():
            self.state = State.CALIBRATED_INTRINSIC

    # reset lại hiệu chuẩn ngoại tham
    def reset_extrinsic_calibration(self) -> None:
        self.state = State.CALIBRATING_EXTRINSIC

    # GIAI ĐOẠN 5: LỌC VÀ THÊM ĐIỂM (FILTERING AND ADDING POINTS)
    def filter_and_add_point(self, point):
        refresh = False
        if point is not None:
            new_point = self.filter.filter(point)
            if new_point is not None:
                self.model.add_point(new_point)
                refresh = True
        else:
            sys.stdout.write('\a')
            sys.stdout.flush()
        return refresh




