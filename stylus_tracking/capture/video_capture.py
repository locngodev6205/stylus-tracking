import cv2


class VideoCapture:
    HEIGHT = 720
    WIDTH = 1280

    def __init__(self, video_source=0, flip_horizontal=False):
        """
        video_source: int (device index, ví dụ 0) hoặc str (URL stream, ví dụ "http://192.168.1.5:4747/video")
        flip_horizontal: True nếu camera bị lật ngang (mirror), ví dụ DroidCam dùng camera trước
        """
        self.flip_horizontal = flip_horizontal
        self.video_capture = cv2.VideoCapture(video_source)
        if not self.video_capture.isOpened():
            raise ValueError("Unable to open video source {}.".format(video_source))

        # Chỉ set resolution khi dùng camera local (device index)
        # IP camera tự quản lý resolution qua app
        if isinstance(video_source, int):
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

    # giải phóng 
    def __del__(self):
        if self.video_capture.isOpened():
            self.video_capture.release()

    def get_next_frame(self) -> (bool, any):
        if self.video_capture.isOpened():
            ret, image = self.video_capture.read()
            if ret:
                # Sửa lỗi camera bị lật ngang (mirror) - phổ biến khi dùng camera trước điện thoại
                if self.flip_horizontal:
                    image = cv2.flip(image, 1)  # 1 = flip theo trục Y (lật ngang)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return ret, image
            else:
                return ret, None
