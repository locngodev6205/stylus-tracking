import math
import numpy as np

class Transform:
    def __init__(self):
        # tạo ma trận đơn vị
        self.matrix = np.eye(4, dtype=np.float32)
    
    def set_translation(self, x, y, z):
        self.matrix[0:3, 3] = [x, y, z]

    def set_rotation(self, x, y, z):
        self.matrix[0:3, 0:3] = self.rodrigues(x, y, z)


    def rodrigues(self, x, y, z):
        # 1. Tính bình phương góc quay và góc quay theta
        theta_sqr = x**2 + y**2 + z**2
        theta = math.sqrt(theta_sqr)

        # 2. Kiểm tra trường hợp đặc biệt: Nếu góc quay gần bằng 0, trả về ma trận đơn vị
        if theta < 1e-12:
            return np.eye(3, dtype=np.float32)

        # 3. Tạo ma trận phản đối xứng (Skew-symmetric matrix)
        # Thay vì tạo từng dòng, ta khởi tạo trực tiếp từ x, y, z
        omega_skew = np.array([
            [ 0, -z,  y],
            [ z,  0, -x],
            [-y,  x,  0]
        ])

        # 4. Tính toán các hệ số chuẩn hóa
        # Công thức: R = I + (sin(theta)/theta) * K + ((1 - cos(theta))/theta^2) * K^2
        sin_theta = math.sin(theta)
        one_minus_cos_theta = 1 - math.cos(theta)
        
        # Hệ số cho omega_skew và omega_skew_sqr
        a = sin_theta / theta
        b = one_minus_cos_theta / theta_sqr

        # 5. Tính ma trận quay cuối cùng
        # omega_skew @ omega_skew là phép nhân ma trận (omega_skew bình phương)
        res = np.eye(3) + a * omega_skew + b * (omega_skew @ omega_skew)

        return res



    def rodrigues_inverse(self, matrix):
        # trace(R) = R11 + R22 + R33
        trace_R = np.trace(matrix)

        # Từ công thức:
        # trace(R) = 1 + 2*cos(theta)
        cos_theta = (trace_R - 1.0) / 2.0

        # Tránh lỗi số học
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Góc quay
        theta = math.acos(cos_theta)

        # ==========================
        # Trường hợp theta ≈ 0
        # ==========================
        if theta < 1e-8:
            return 0.0, 0.0, 0.0

        # ==========================
        # Trường hợp thông thường
        # 0 < theta < pi
        # ==========================
        if abs(theta - math.pi) > 1e-6:
            # R−RT = 2sin(θ)K
            ux = (matrix[2, 1] - matrix[1, 2]) / (2.0 * math.sin(theta))
            uy = (matrix[0, 2] - matrix[2, 0]) / (2.0 * math.sin(theta))
            uz = (matrix[1, 0] - matrix[0, 1]) / (2.0 * math.sin(theta))

            x = theta * ux
            y = theta * uy
            z = theta * uz

            return x, y, z

        # chưa hiểu
        # ==========================
        # Trường hợp theta ≈ pi
        # R=I+2K^2 => R=2uuT−I
        # ==========================
        # Lấy trục quay từ đường chéo chính

        ux = math.sqrt(max(0.0, (matrix[0, 0] + 1.0) / 2.0))
        uy = math.sqrt(max(0.0, (matrix[1, 1] + 1.0) / 2.0))
        uz = math.sqrt(max(0.0, (matrix[2, 2] + 1.0) / 2.0))

        axis = np.array([ux, uy, uz])

        norm = np.linalg.norm(axis)

        if norm > 1e-8:
            axis /= norm

        rotation_vector = math.pi * axis

        return tuple(rotation_vector)


    def combine(self, transform, copy=False):
        if copy:
            new_matrix = self.matrix @ transform.matrix
            return Transform.from_matrix(new_matrix)

        self.matrix = self.matrix @ transform.matrix
        return self

    @staticmethod
    def from_matrix(matrix):
        ret = Transform()
        ret.matrix = matrix
        return ret

    # với điều kiện matrix self là kết hợp chỉ tịnh tiến và quay
    def inverse(self):
        ret = Transform()
        ret.matrix[0:3, 0:3] = self.matrix[0:3, 0:3].transpose() # R new
        ret.matrix[0:3, 3] = -ret.matrix[0:3, 0:3] @ self.matrix[0:3, 3] # t new 

        return ret

    def dot(self, points):
        shape = points.shape
        if shape[1] == 3:
            ones = np.ones((shape[0], 1))
            homogeneous = np.hstack((points, ones)).T
        elif shape[1] == 4:
            homogeneous = points.T
        else:
            raise ValueError(
                "input array has to be of size 3 or in homogeneous coordinate, current size = " + str(shape))
        return (self.matrix @ homogeneous).T[:, 0:3]
        
    def to_parameters(self, isDegree=False):
        x, y, z = self.matrix[0:3, 3]
        a, b, c = self.rodrigues_inverse(self.matrix[0:3, 0:3])
        if isDegree:
            a = math.degrees(a)
            b = math.degrees(b)
            c = math.degrees(c)
        ret = [x, y, z, a, b, c]

        return np.array(ret)