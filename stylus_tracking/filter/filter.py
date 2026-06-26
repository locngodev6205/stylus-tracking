from collections import deque
import numpy as np
import time


class FilterNone:
    def __init__(self):
        pass

    def filter(self, point):
        return point


class FilterMedian:
    BUFFER_SIZE = 9

    def __init__(self):
        self.buffer = deque([], self.BUFFER_SIZE)

    def filter(self, point):
        self.buffer.append(np.array(point))
        if len(self.buffer) == self.BUFFER_SIZE:
            vals = np.array(list(self.buffer))
            return np.median(vals, axis=0)
        else:
            return None


class FilterKalman:
    '''
    Kalman filter với mô hình Constant Acceleration (Gia tốc không đổi).
    State vector 9 chiều: [x, y, z, vx, vy, vz, ax, ay, az]
    
    So với bản cũ (Constant Velocity, 6 chiều):
    - Thêm gia tốc (ax, ay, az) giúp bám theo bút tốt hơn khi tăng/giảm tốc
    - Thêm hàm predict() để dự đoán tọa độ TƯƠNG LAI (Trajectory Forecasting)
    '''

    DIM = 9  # Số chiều state vector

    def __init__(self):
        self.prev_time = 0
        self.prev_point = np.array([0, 0, 0], dtype=float)
        self.prev_velocity = np.array([0, 0, 0], dtype=float)
        self.dt = 0.033  # dt mặc định (~30 FPS)

        # State vector: [x, y, z, vx, vy, vz, ax, ay, az]
        self.x = np.matrix(np.zeros(self.DIM)).T

        # Ma trận độ tự tin (Covariance Matrix), số càng lớn bộ lọc càng thấy mình đoán sai
        self.P = np.matrix(np.eye(self.DIM) * 1.0, dtype=float)

        self.q = 0.5   # Bút di chuyển thất thường cỡ nào (tăng lên vì có thêm gia tốc)
        self.r = 0.1   # Camera đo sai cỡ nào

        # Ma trận (sẽ được khởi tạo bên dưới)
        self.A = np.matrix([])  # State transition (thay đổi theo dt)
        self.H = np.matrix([])  # Measurement (hằng số)
        self.Q = np.matrix([])  # Process noise (hằng số)
        self.R = np.matrix([])  # Sensor noise (hằng số)

        self.init_constant_matrices()
        self.init_matrix_depending_on_dt(dt=0.2)

    def filter(self, measured_point):
        current_time = time.time()  # TODO use time of img capture
        dt = current_time - self.prev_time
        if dt <= 0 or dt > 1.0:  # Bảo vệ: dt quá lớn (>1s) → reset
            dt = 0.033
        self.dt = dt
        self.init_matrix_depending_on_dt(dt)

        x = self.x
        P = self.P
        I = np.matrix(np.eye(self.DIM))
        A = self.A
        H = self.H
        Q = self.Q
        R = self.R

        # measurement vector
        measured_point, m = self.measurement(measured_point, dt)

        # prediction step (Dự đoán)
        x = A * x
        P = Q + (A * P * A.T)

        # correction step (Sửa sai bằng dữ liệu camera)
        S = R + (H * P * H.T)
        K = P * H.T * S.I
        y = m - (H * x)

        self.x = x + (K * y)
        self.P = (I - (K * H)) * P

        new_point = np.array(self.x[0:3].T)[0, :]
        self.prev_point = new_point
        self.prev_time = current_time
        return np.append(new_point, [1])  # cart 2 hom

    def predict(self, steps_ahead=1):
        '''
        Dự đoán vị trí bút SAU 'steps_ahead' khung hình,
        mà KHÔNG CẦN đợi dữ liệu camera.

        Ví dụ: steps_ahead=2 → dự đoán tọa độ sau 2 frame (~66ms)
        Trả về: np.array [x, y, z] tọa độ tương lai
        '''
        dt_future = self.dt * steps_ahead
        A_future = self._build_A(dt_future)

        # Nhân ma trận để "nhảy cóc" về tương lai
        x_future = A_future * self.x

        return np.array(x_future[0:3].T)[0, :]

    def _build_A(self, dt):
        '''Tạo ma trận State Transition cho khoảng thời gian dt bất kỳ'''
        dt2 = 0.5 * dt * dt
        return np.matrix([
            # x        y        z        vx       vy       vz       ax       ay       az
            [1,       0,       0,       dt,      0,       0,       dt2,     0,       0      ],  # x  = x + vx*dt + 0.5*ax*dt²
            [0,       1,       0,       0,       dt,      0,       0,       dt2,     0      ],  # y  = y + vy*dt + 0.5*ay*dt²
            [0,       0,       1,       0,       0,       dt,      0,       0,       dt2    ],  # z  = z + vz*dt + 0.5*az*dt²
            [0,       0,       0,       1,       0,       0,       dt,      0,       0      ],  # vx = vx + ax*dt
            [0,       0,       0,       0,       1,       0,       0,       dt,      0      ],  # vy = vy + ay*dt
            [0,       0,       0,       0,       0,       1,       0,       0,       dt     ],  # vz = vz + az*dt
            [0,       0,       0,       0,       0,       0,       1,       0,       0      ],  # ax = ax (gia tốc không đổi)
            [0,       0,       0,       0,       0,       0,       0,       1,       0      ],  # ay = ay
            [0,       0,       0,       0,       0,       0,       0,       0,       1      ],  # az = az
        ], dtype=float)

    def init_matrix_depending_on_dt(self, dt=0.1):
        self.A = self._build_A(dt)

    def init_constant_matrices(self):
        # Ma trận Measurement H (9x9): Camera đo được vị trí và vận tốc
        self.H = np.matrix(np.zeros((self.DIM, self.DIM)), dtype=float)
        self.H[0, 0] = 1  # Đo x
        self.H[1, 1] = 1  # Đo y
        self.H[2, 2] = 1  # Đo z
        self.H[3, 3] = 1  # Đo vx
        self.H[4, 4] = 1  # Đo vy
        self.H[5, 5] = 1  # Đo vz

        # Ma trận Process Noise Q (9x9): Nhiễu chủ yếu đến từ gia tốc thay đổi bất ngờ
        self.Q = np.matrix(np.zeros((self.DIM, self.DIM)), dtype=float)
        self.Q[6, 6] = 1  # ax thay đổi bất ngờ
        self.Q[7, 7] = 1  # ay thay đổi bất ngờ
        self.Q[8, 8] = 1  # az thay đổi bất ngờ
        self.Q *= self.q

        # Ma trận Sensor Noise R (9x9)
        self.R = np.matrix(np.eye(self.DIM), dtype=float) * self.r

    # hàm tính vận tốc đo được
    def measurement(self, point, dt):
        point = np.array(point[0:3])
        dPoint = point - self.prev_point  # Quãng đường (displacement)
        if dt > 0:
            velocity = dPoint / dt  # Vận tốc = Quãng đường / Thời gian (mm/s)
            acceleration = (velocity - self.prev_velocity) / dt  # Gia tốc = ΔV / Δt (mm/s²)
        else:
            velocity = np.zeros(3)
            acceleration = np.zeros(3)
        self.prev_velocity = velocity
        # Measurement vector: [x, y, z, vx, vy, vz, 0, 0, 0]
        # (Camera không đo được gia tốc trực tiếp, chỉ suy ra từ vận tốc)
        return point, np.matrix(np.concatenate([point, velocity, np.zeros(3)], axis=None)).T


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     kalman_filter = FilterKalman()
#     # kalman_filter = FilterMedian()

#     n_iter = 50
#     sz = (n_iter,)  # size of array

#     # allocate space for arrays
#     xhat = np.zeros(sz)  # a posteri estimate of x
#     yhat = np.zeros(sz)

#     start_x = 20  # TODO try 20
#     start_y = 20  # TODO try 20

#     xtrue = np.add.accumulate(np.linspace(start_x, 100, n_iter))
#     ytrue = np.linspace(start_y, 125, n_iter)
#     xsensor = xtrue + np.random.normal(0, 0.8, size=sz)
#     ysensor = ytrue + np.random.normal(0, 0.8, size=sz)

#     for k in range(1, n_iter):
#         time.sleep(0.05)
#         point = np.array([xsensor[k], ysensor[k], 0])
#         new_point = kalman_filter.filter(point)
#         if new_point is None:
#             new_point = (0, 0, 0)
#         xhat[k], yhat[k], _ = new_point

#     plt.figure()
#     plt.plot(xsensor, 'k+', label='measurement (kalman input)')
#     plt.plot(xhat, 'b-', label='kalman position estimate')
#     plt.plot(xtrue, color='g', label='truth')
#     plt.plot(ysensor, 'k+', label='measurement (kalman input)')
#     plt.plot(yhat, 'b-', label='kalman position estimate')
#     plt.plot(ytrue, color='g', label='truth')
#     plt.legend()
#     plt.title('Estimate vs. iteration step', fontweight='bold')
#     plt.xlabel('Iteration')
#     plt.ylabel('Voltage')
#     plt.show()

