from enum import Enum, auto
from logging import Logger

import cv2
import numpy as np
from cv2 import aruco

# Mục đích: Tìm các thông số kỹ thuật bên trong của camera
# Tiêu cự, tâm ảnh, các hệ số méo của thấu kịnh

class State(Enum):
    RAW = auto()
    CALIBRATING_INTRINSIC = auto()
    CALIBRATED_INTRINSIC = auto()
    CALIBRATING_EXTRINSIC = auto()
    CALIBRATED = auto()


class Calibration:
    INTRINSIC_PARAMETERS_FILENAME = "/mnt/windows/Users/62205/Documents/locngo/self-study/IOT/Stylus Tracking/stylus-tracking/stylus_tracking/intrinsic_parameters.npz"
    ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    ARUCO_PARAMETERS = aruco.DetectorParameters()

    CORNERS_IDS = (100, 101, 102, 103)
    REQUIRED_FRAMES = 30  # Tăng số frame để calib chính xác hơn (trước đây là 10)

    # Bản vẽ hiệu chuẩn ngoại tham (OBJECT_POINTS), góc ở tâm bản vẽ
    # Dùng để:
    #   - Tính toán: Là "mỏ neo" để giải các bài toán vật lý, hình học (như góc nghiêng, khoảng cách).
    #   - Lập trình: Giúp lập trình viên xác định tâm (origin), hướng và tỉ lệ của hệ tọa độ.
    # Center as origin (0, 0, 0)
    OBJECT_POINTS = np.zeros((4, 3))
    OBJECT_POINTS[0] = [-101.45, -133.2, 0]
    OBJECT_POINTS[1] = [101.45, -133.2, 0]
    OBJECT_POINTS[2] = [101.45, 133.2, 0]
    OBJECT_POINTS[3] = [-101.45, 133.2, 0]

    def __init__(self, logger: Logger):
        self.logger = logger

        self.rvecs = None  # rotation vectors
        self.tvecs = None  # translation vectors
        self.intrinsic_parameters = None  # intrinsic parameters
        self.state = State.RAW  # state

        self.criteria = None  # termination criteria
        self.objp = None # Object Points: Tọa độ 3D lý tưởng (10x7 ô cờ), shape (70, 3).
        self.objpoints = None # [M, 70, 3] : Danh sách các bộ Object Points từ M frame.
        self.imgpoints = None # [M, 70, 1, 2] : Danh sách các tọa độ góc thực tế tìm thấy trong ảnh từ M frame.
        self.valid_frames = None # Đếm số frame đã valid (tối thiểu 10 frame )
        self.frame_counter = None # Đếm số frame (tối thiểu 30 frame )

    def try_load_intrinsic(self) -> bool:
        try:
            self.logger.info("Trying to retrieve last intrinsic calibration parameters.")
            self.intrinsic_parameters = np.load(self.INTRINSIC_PARAMETERS_FILENAME, allow_pickle=True)['intrinsic_parameters'].item(0)
            self.logger.debug(type(self.intrinsic_parameters), self.intrinsic_parameters)
        except IOError:
            self.logger.info("Could not load previous intrinsic parameters.")
            return False
        self.logger.info("Loaded previous INTRINSIC parameters successfully.")
        self.logger.info(f" -> cameraMatrix:\n{self.intrinsic_parameters['cameraMatrix']}")
        self.logger.info(f" -> distCoef: {self.intrinsic_parameters['distCoef'].flatten()}")
        return True

    # 
    def calculate_extrinsic(self, frame) -> bool:
        self.logger.info("Starting EXTRINSIC calibration.")
        # quét tìm 4 marker ở bàn
        _, corners, ids = self.get_frame_with_aruco_label(frame)

        # === DEBUG: In ra các marker đang thấy ===
        if ids is not None and len(ids) > 0:
            detected_ids = ids.flatten().tolist()
            self.logger.info(f"  Detected marker IDs: {detected_ids}")
            missing = [cid for cid in self.CORNERS_IDS if cid not in detected_ids]
            if missing:
                self.logger.warning(f"  Missing corner IDs: {missing} (cần tất cả {list(self.CORNERS_IDS)})")
        else:
            self.logger.warning("  No markers detected at all! Hãy chỉa camera vào 4 marker góc bàn.")

        # trích xuất tọa độ 2D của 4 marker
        if np.any(ids) and np.all(np.isin(self.CORNERS_IDS, ids)):
            image_points = np.zeros((4, 2))
            for id, corner in zip(ids, corners):
                index = self.CORNERS_IDS.index(id)
                image_points[index] = corner[0, index, :]

            # Giải bài toán hình học không gian 3D
            _, rvec, tvec = cv2.solvePnP(self.OBJECT_POINTS, image_points,
                                                     self.intrinsic_parameters['cameraMatrix'],
                                                     self.intrinsic_parameters['distCoef'])
            self.rvecs = rvec
            self.tvecs = tvec
            self.logger.info("Extrinsic calibration calculated successfully.")
            self.logger.info(f" -> tvec (Translation): {tvec.flatten()}")
            self.logger.info(f" -> rvec (Rotation):    {rvec.flatten()}")
            return True
        self.logger.error("Extrinsic calibration could not be completed.")
        return False

    # Tính toán camera matrix, distCoef
    # camera matrix, distCoef là các tham số nội tại của camera
    def calculate_intrinsic(self, frame) -> bool:
        # Đợi 30 frame đầu
        if self.frame_counter < 30:
            self.frame_counter += 1
            return False

        # Đếm số frame đã valid
        elif self.valid_frames < self.REQUIRED_FRAMES:
            self.frame_counter = 0
            self.logger.info(f"Show checkerboard for intrinsic calibration: {self.valid_frames}/{self.REQUIRED_FRAMES}.")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # corners: Danh sách tọa độ 2D của 70 góc tìm thấy.
            ret, corners = cv2.findChessboardCorners(frame, (10, 7), None)  # Checkerboard 11x8 (23mmx23mm) each
            if ret:
                self.valid_frames += 1
                # sub-pixel: để tăng độ chính xác của các góc tìm thấy
                corners2 = cv2.cornerSubPix(frame, corners, (11, 11), (-1, -1), self.criteria)
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)
        if self.valid_frames >= self.REQUIRED_FRAMES:
            # input:
            #   self.objpoints: tọa độ 3D thực tế lý tưởng của các góc ô cờ
            #   self.imgpoints: tọa độ 2D của các góc ô cờ trong ảnh
            #   frame.shape[::-1]: kích thước ảnh
            # output:
            #   ret: tổng bình phương các phần dư tối thiểu hóa
            #   cameraMatrix: ma trận camera
            #   distCoef: hệ số méo
            #   rvecs: vector quay
            #   tvecs: vector tịnh tiến
            ret, cameraMatrix, distCoef, rvecs, tvecs = cv2.calibrateCamera(self.objpoints,
                                                                            self.imgpoints,
                                                                            frame.shape[::-1],
                                                                            None,
                                                                            None)
            self.intrinsic_parameters = {
                'cameraMatrix': cameraMatrix,
                'distCoef': distCoef,
                # 'rvecs': self.rvecs, 'tvecs': self.tvecs,
            }
            self.save_intrinsic()
            self.logger.info("Intrinsic calibration calculated and saved successfully.")
            self.logger.info(f" -> cameraMatrix:\n{cameraMatrix}")
            self.logger.info(f" -> distCoef: {distCoef.flatten()}")
            return True
        return False

    def start_intrinsic_calibration(self) -> None:
        # điều khiển vòng lặp của thuật toán tìm kiếm góc
        # vong lăp đủ 30 hoặc sai số nhỏ hơn 0.001
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Dựng lưới tọa độ 3D mẫu cho bàn cờ
        # giống như (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((10 * 7, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)

        # xóa sạch dữ liệu lưu lại dữ liệu mới
        self.objpoints = []  # Lưu tọa độ 3D thực tế (của bàn cờ)
        self.imgpoints = []  # Lưu tọa độ 2D tìm thấy trong ảnh
        self.valid_frames = 0 # Đếm số frame đã valid (tối thiểu 10 frame)
        self.frame_counter = 0 # Đếm số frame (tối thiểu 30 frame)
        self.logger.info("Starting INTRINSIC calibration.")

    # Lưu dữ liệu intrinsic_parameters vào file .npz
    def save_intrinsic(self) -> None:
        np.savez(self.INTRINSIC_PARAMETERS_FILENAME, intrinsic_parameters=self.intrinsic_parameters)

    # Phát diện các mã aruco ở 4 góc bàn (từ đó giúp biến được có đúng với 4 marker đã lưu ko)
    def get_frame_with_aruco_label(self, image):
        image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Easily print aruco markers here: http://chev.me/arucogen/
        corners, ids, rejected_img_points = aruco.detectMarkers(image_grey,
                                                                self.ARUCO_DICT, parameters=self.ARUCO_PARAMETERS)

        if np.any(ids):
            self.logger.debug(ids)
        if np.any(corners):
            self.logger.debug(corners)

        img_color_labeled = aruco.drawDetectedMarkers(image, corners)
        return img_color_labeled, corners, ids
