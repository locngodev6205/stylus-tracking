import math
import numpy as np
import cv2
from cv2 import aruco

from stylus_tracking.calibration import calibration
from stylus_tracking.detection import transform

PENCIL_LENGTH = 145  # [mm] from dodecahedron center to tip of pencil.


# Bài toán đặt ra là tìm được tọa độ đầu bút so với mặt phẳng của bàn


class Detection:

    def __init__(self, cam_param: calibration):
        # Thông số camera
        self.cam_param = cam_param 
        self.success = False
        # Tải bộ từ điển Aruco 4x4, 50 mẫu
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        # Tham số nhận dạng Aruco (tham số mặc định)
        self.parameters = aruco.DetectorParameters()

        # Tạo bàn Aruco với 12 điểm đánh dấu và 12 ID tương ứng
        points = dodecahedron_aruco_points() # Tọa độ 12 marker (top left, top right, bot right, bot left)
        ids = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]) # ID tương ứng với 12 điểm đánh dấu
        self.board = aruco.Board(points, self.marker_dict, ids)

        

        # Mô tả mối quan hệ từ tâm mặt bàn đến tâm camera (World-to-Camera)
        world_to_camera = transform.Transform.from_parameters(
                                            self.cam_param.tvecs[0].item(), # mảng một chiều sang giá trị vô hướng
                                            self.cam_param.tvecs[1].item(),
                                            self.cam_param.tvecs[2].item(),
                                            self.cam_param.rvecs[0].item(),
                                            self.cam_param.rvecs[1].item(),
                                            self.cam_param.rvecs[2].item())

        self.camera_to_world = world_to_camera.inverse()

        # Tính toán vector tịnh tiến từ tâm đến đầu bút
        self.tvec_pencil = pencil_tip_from_length_mm(PENCIL_LENGTH)
        tp = self.tvec_pencil
        # Mô tả mối quan hệ từ đầu bút đến tâm khối 12 mặt (Tip-to-Stylus)
        # trả về ma trận đồng nhất 4x4
        self.tip_to_stylus = transform.Transform.from_parameters(tp[0].item(), tp[1].item(), tp[2].item(), 0, 0, 0)

    # bước 1: Nhận dạng
    def detect(self, img):
        # chuyển hình ảnh sang grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Bước 1.1: Nhận dạng các ArUco Marker
        # corners: tọa độ các góc của marker, góc tọa độ ở top-left khung hình camera
        # id marker theo thứ tự
        # hình vuông bị loại bỏ
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.marker_dict, parameters=self.parameters)

        # === LUÔN VẼ TẤT CẢ MARKER ĐƯỢC PHÁT HIỆN (kể cả khi chưa đủ để ước lượng tư thế) ===
        num_detected = 0
        if ids is not None and len(ids) > 0:
            num_detected = len(ids)
            img = aruco.drawDetectedMarkers(img, corners, ids)

        # Bước 1.2: Ước lượng tư thế của các ArUco Marker
        # Kiểm tra xem có phát hiện marker nào không trước khi ước lượng tư thế
        # OpenCV 4.13 sẽ crash nếu ids rỗng
        if ids is not None and len(ids) > 0:
            rvec_init = np.zeros((3, 1), dtype=np.float64)
            tvec_init = np.zeros((3, 1), dtype=np.float64)
            num_markers, rotation, translation_ = aruco.estimatePoseBoard(
                corners, ids, self.board,
                self.cam_param.intrinsic_parameters['cameraMatrix'],
                self.cam_param.intrinsic_parameters['distCoef'],
                rvec_init, tvec_init)
            self.success = num_markers > 0
        else:
            self.success = False

        if self.success:
            rvec = rotation.copy()
            tvec = translation_.copy()

            # # Vẽ hệ trục tọa độ, 3 trục 3 màu (X: đỏ, Y: lục, Z: lam)
            # img = cv2.drawFrameAxes(img, self.cam_param.intrinsic_parameters['cameraMatrix'],
            #                        self.cam_param.intrinsic_parameters['distCoef'], rvec, tvec, length=100)

            # Tạo đối trượng Transform chứa tất cả thông tin biến đổi (Stylus-to-Camera)
            stylus_to_camera = transform.Transform.from_parameters(tvec[0].item(), tvec[1].item(),
                                                         tvec[2].item(), rvec[0].item(),
                                                         rvec[1].item(), rvec[2].item())

            # Mô tả mối quan hệ biến đổi từ đầu bút đến camera (Tip-to-Camera)
            tip_to_camera = stylus_to_camera.combine(self.tip_to_stylus, True)

            # Mô tả mối quan hệ biến đổi từ đầu bút đến thế giới (Tip-to-World)
            tip_to_world = self.camera_to_world.combine(tip_to_camera, True)

            tip_info = tip_to_world.to_parameters(True)
            # Tọa độ đầu bút
            position_x = tip_info[0]
            position_y = tip_info[1]
            position_z = tip_info[2]

            # === VẼ TỌA ĐỘ VÀ CHẤM ĐỎ NGAY TẠI ĐẦU BÚT TRÊN ẢNH CAMERA ===
            tip_3d_cam = tip_to_camera.matrix[0:3, 3] # vị trí 3D của đầu bút so với camera
            img_pts, _ = cv2.projectPoints(
                np.array([tip_3d_cam], dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                self.cam_param.intrinsic_parameters['cameraMatrix'],
                self.cam_param.intrinsic_parameters['distCoef']
            )
            pixel_x = int(img_pts[0][0][0])
            pixel_y = int(img_pts[0][0][1])

            # Vẽ chấm đỏ và vòng tròn trắng bao quanh tại vị trí đầu bút trên camera
            cv2.circle(img, (pixel_x, pixel_y), 6, (0, 0, 255), -1)
            cv2.circle(img, (pixel_x, pixel_y), 10, (255, 255, 255), 2)
            # Viết tọa độ ngay cạnh đầu bút trên màn hình camera
            cv2.putText(img, f"Tip: ({position_x:.1f}, {position_y:.1f}, {position_z:.1f})", 
                        (pixel_x + 15, pixel_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # # === VẼ THÔNG TIN LÊN MÀN HÌNH (OVERLAY) ===
            # h, w = img.shape[:2]
            # # Nền bán trong suốt cho phần text
            # overlay = img.copy()
            # cv2.rectangle(overlay, (10, 10), (420, 130), (0, 0, 0), -1)
            # cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

            # # Trạng thái nhận diện
            # cv2.putText(img, f"Markers: {num_detected} detected", (20, 35),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # # Tọa độ đầu bút
            # cv2.putText(img, f"X: {position_x:7.1f} mm", (20, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # cv2.putText(img, f"Y: {position_y:7.1f} mm", (20, 85),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # # Màu cho Z: xanh lá nếu gần bàn (|z| < 20), vàng nếu hơi cao, đỏ nếu quá cao
            # if abs(position_z) < 20:
            #     z_color = (0, 255, 0)   # Xanh lá - gần mặt bàn
            # elif abs(position_z) < 80:
            #     z_color = (0, 255, 255) # Vàng - hơi cao
            # else:
            #     z_color = (0, 0, 255)   # Đỏ - quá cao / nhảy bất thường
            # cv2.putText(img, f"Z: {position_z:7.1f} mm", (20, 110),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, z_color, 2)

            # # In ra terminal (giữ nguyên)
            # print(f"\r Tip position: x={position_x:7.1f}  y={position_y:7.1f}  z={position_z:7.1f} mm", end="")

            return img, (position_x, position_y, position_z, 1)
        else:
            # # === KHI KHÔNG NHẬN DIỆN ĐƯỢC TƯ THẾ ===
            # overlay = img.copy()
            # cv2.rectangle(overlay, (10, 10), (420, 60), (0, 0, 0), -1)
            # cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

            # if num_detected > 0:
            #     cv2.putText(img, f"Markers: {num_detected} (pose failed)", (20, 40),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            # else:
            #     cv2.putText(img, "No markers detected", (20, 40),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            return img, None


def rotation_around_y(d):
    r = np.deg2rad(d)
    return np.matrix(
        [
            [np.cos(r), 0, -np.sin(r), 0], 
            [0, 1, 0, 0], 
            [np.sin(r), 0, np.cos(r), 0], 
            [0, 0, 0, 1]
        ],
        dtype=np.float32)


def rotation_around_z(d):
    r = np.deg2rad(d)
    return np.matrix(
        [
            [np.cos(r), np.sin(r), 0, 0], 
            [-np.sin(r), np.cos(r), 0, 0], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ],
        dtype=np.float32)

# chưa
def to_homogenous_position(a):
    size = a.shape[0]
    res = np.ones((size+1, 1))
    res[:size, :] = a
    return res
# chưa
def to_homogenous_translation(a):
    size = a.shape[0]
    res = np.identity(size+1)
    res[:size,size] = a.flatten()
    return res
# chưa
def to_homogenous_rotation(a):
    size = a.shape[0]
    res = np.identity(size+1)
    res[:size,:size] = a
    return res

def translation(tx, ty, tz):
    return np.matrix(
        [
            [1, 0, 0, tx], 
            [0, 1, 0, ty], 
            [0, 0, 1, tz], 
            [0, 0, 0, 1]
        ],
        dtype=np.float32)

# Homogeneous to Cartesian (4D to 3D)
# [x, y, z, w] -> [x/w, y/w, z/w]
def hom2cart(p):
    return p[:-1] / p[-1]

# Tính toán tọa độ 12 marker Aruco trên hình đa diện đều, gốc ở tâm đa diện
def dodecahedron_aruco_points() -> np.array:
    radius = 25  # mm bán kính hình đa diện
    tc = 25 / 3  # mm dịch chuyển x và y của các góc so với tâm mặt
    all_aruco_points = []
    # 4 góc của mỗi marker
    origin_points = np.matrix([ [-tc, -tc, 0, 1], # top-left
                                [-tc, tc, 0, 1], # top-right
                                [tc, tc, 0, 1], # bottom-right
                                [tc, -tc, 0, 1] # bottom-left
                            ], dtype=np.float32).T


    # first row
    for i in [2, 1, 0, 4, 3]:
        aruco_corners = rotation_around_z(72 * i) * rotation_around_y(116.565) * \
                        rotation_around_z(180) * translation(0, 0, radius) * origin_points
        all_aruco_points.append(hom2cart(aruco_corners).T)
    # second row
    for i in [0, 4, 3, 2, 1]:
        aruco_corners = rotation_around_z(72 * i) * rotation_around_y(116.565) * \
                        translation(0, 0, -radius) * rotation_around_y(180) * origin_points
        all_aruco_points.append(hom2cart(aruco_corners).T)

    # top
    aruco_corners = translation(0, 0, radius) * origin_points
    all_aruco_points.append(hom2cart(aruco_corners).T)
    # bottom
    aruco_corners = translation(0, 0, -radius) * rotation_around_y(180) * origin_points
    all_aruco_points.append(hom2cart(aruco_corners).T)

    all_aruco_points = np.array(all_aruco_points, dtype=np.float32)
    return all_aruco_points

# trả về tọa đổ tip, góc là góc của khối 12 mặt
def pencil_tip_from_length_mm(pencil_length):
    zero_point = np.array([[0], [0], [0], [1]])
    pencil_tip = rotation_around_y(116.565 / 3) * translation(0, 0, -pencil_length) * zero_point
    return pencil_tip
