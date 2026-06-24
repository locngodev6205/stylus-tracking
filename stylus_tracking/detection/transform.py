import math
import numpy as np

class Transform:
    def __init__(self):
        # tạo ma trận đơn vị
        self.matrix = np.eye(4, dtype=np.float32)
    
    def set_translation(self, x, y, z):
        self.matrix[0:3, 3] = [x, y, z]

    def set_rotaion(self, x, y, z):
        self.matrix[0:3, 0:3] = self.rodrigues(x, y, z)

    def rodrigues(self, x, y, z):
        matrix = np.eye(3)
        

    