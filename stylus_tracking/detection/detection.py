import math
import numpy as np
import cv2
from cv2 import aruco

from stylus_tracking.calibration import calibration
from stylus_tracking.detection import transform

PENCIL_LENGTH = 153  # [mm] from dodecahedron center to tip of pencil.


class Detection:

    def __init__(self, cam_param: calibration):
        # Thông số camera
        self.cam_param = cam_param 
        self.success = False
        # Tải bộ từ điển Aruco 4x4, 50 mẫu
        self.marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        # Tham số nhận dạng Aruco (tham số mặc định)
        self.parameters = aruco.DetectorParameters_create()

        # Tạo bàn Aruco với 12 điểm đánh dấu và 12 ID tương ứng
        points = dodecahedron_aruco_points() # Tọa độ 12 marker (top left, top right, bot right, bot left)
        ids = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]) # ID tương ứng với 12 điểm đánh dấu
        self.board = aruco.Board_create(points, self.marker_dict, ids)

        

        # Mô tả mối quan hệ từ tâm mặt bàn đến tâm camera (World-to-Camera)
        world_to_camera = transform.Transform.from_parameters(np.asscalar(self.cam_param.tvecs[0]),
                                            np.asscalar(self.cam_param.tvecs[1]),
                                            np.asscalar(self.cam_param.tvecs[2]),
                                            np.asscalar(self.cam_param.rvecs[0]),
                                            np.asscalar(self.cam_param.rvecs[1]),
                                            np.asscalar(self.cam_param.rvecs[2]))

        self.camera_to_world = world_to_camera.inverse()

        # Tính toán vector tịnh tiến từ tâm đến đầu bút
        self.tvec_pencil = pencil_tip_from_length_mm(PENCIL_LENGTH)
        tp = self.tvec_pencil
        # Mô tả mối quan hệ từ đầu bút đến tâm khối 12 mặt (Tip-to-Stylus)
        # trả về ma trận đồng nhất 4x4
        self.tip_to_stylus = transform.Transform.from_parameters(tp[0], tp[1], tp[2], 0, 0, 0)

    # bước 1: Nhận dạng
    def detect(self, img):
        # chuyển hình ảnh sang grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



        # Bước 1.1: Nhận dạng các ArUco Marker
        # corners: tọa độ các góc của marker, góc tọa độ ở top-left khung hình camera
        # id marker theo thứ tự
        # hình vuông bị loại bỏ
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.marker_dict, parameters=self.parameters,
                                                              cameraMatrix=self.cam_param.intrinsic_parameters[
                                                                  'cameraMatrix'],
                                                              distCoeff=self.cam_param.intrinsic_parameters['distCoef'])

        # Bước 1.2: Ước lượng tư thế của các ArUco Marker
        # success: trạng thái true, false
        # rvec: [rx, ry, rz]
        # tvec: [tx, ty, tz]
        self.success, rotation, translation_ = aruco.estimatePoseBoard(corners, ids, self.board,
                                                                      self.cam_param.intrinsic_parameters[
                                                                          'cameraMatrix'],
                                                                      self.cam_param.intrinsic_parameters[
                                                                          'distCoef'])

        if self.success:
            rvec = rotation.copy()
            tvec = translation_.copy()

            img = aruco.drawDetectedMarkers(img, corners, ids)

            img = aruco.drawAxis(img, self.cam_param.intrinsic_parameters['cameraMatrix'],
                                   self.cam_param.intrinsic_parameters['distCoef'], rvec, tvec, length=100)

            #print(rvec)
            #print(tvec)


            # Tạo đối trượng Transform chứa tất cả thông tin biến đổi (Stylus-to-Camera)
            # Tính toán nó thành hai mà trận 3x3 (ma trận xoay) và 3x1 (ma trận tịnh tiến) 
            # Kết quả ra ma trận 4x4 (dạng đồng nhất) -> Extrinsic Matrix
            stylus_to_camera = transform.Transform.from_parameters(np.asscalar(tvec[0]), np.asscalar(tvec[1]),
                                                         np.asscalar(tvec[2]), np.asscalar(rvec[0]),
                                                         np.asscalar(rvec[1]), np.asscalar(rvec[2]))

            # Mô tả mối quan hệ biến đổi từ đầu bút đến camera (Tip-to-Camera)
            tip_to_camera = stylus_to_camera.combine(self.tip_to_stylus, True)
            # TODO return position + orientation of the stylus


            # Mô tả mối quan hệ biến đổi từ đầu bút đến thế giới (Tip-to-World)
            tip_to_world = self.camera_to_world.combine(tip_to_camera, True)

            tip_info = tip_to_world.to_parameters(True)
            position_x = tip_info[0]
            position_y = tip_info[1]
            position_z = tip_info[2]



            return img, (position_x, position_y, position_z, 1)
        else:
            return img, None


# TODO new file
def rotation_around_y(d):
    r = np.deg2rad(d)
    return np.matrix([[np.cos(r), 0, -np.sin(r), 0], [0, 1, 0, 0], [np.sin(r), 0, np.cos(r), 0], [0, 0, 0, 1]],
                     dtype=np.float32)


def rotation_around_z(d):
    r = np.deg2rad(d)
    return np.matrix([[np.cos(r), np.sin(r), 0, 0], [-np.sin(r), np.cos(r), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                     dtype=np.float32)


def to_homogenous_position(a):
    size = a.shape[0]
    res = np.ones((size+1, 1))
    res[:size, :] = a
    return res

def to_homogenous_translation(a):
    size = a.shape[0]
    res = np.identity(size+1)
    res[:size,size] = a.flatten()
    return res

def to_homogenous_rotation(a):
    size = a.shape[0]
    res = np.identity(size+1)
    res[:size,:size] = a
    return res

def translation(tx, ty, tz):
    return np.matrix([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]], dtype=np.float32)

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
                        rotation_around_y(180) * translation(0, 0, -radius) * origin_points
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
