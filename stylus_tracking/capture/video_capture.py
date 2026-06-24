# pyrefly: ignore [missing-import]
import cv2
from typing import Tuple, Any

class VideoCapture:
    HEIGHT = 720
    WIDTH = 1280

    def __init__(self, video_source=0, flip_horizontal=False):
        self.flip_horizontal = flip_horizontal
        self.video_capture = cv2.VideoCapture(video_source)
        if not self.video_capture.isOpened():
            raise ValueError("Unable to open vdieo source {}.".format(video_source))

        # Kiểm tra vidoe_source có phải là int hay không
        if isinstance(video_source, int):
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

    def __del__(self):
        if self.video_capture.isOpened():
            self.video_capture.release()

    def get_next_frame(self) -> Tuple[bool, Any]:
        if self.video_capture.isOpened():
            ret, image = self.video_capture.read()
            if ret:
                if self.flip_horizontal:
                    image = cv2.flip(image, 1) # flip theo truc Y
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return ret, image
            else:
                return ret, image

