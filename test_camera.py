import cv2
import sys
# Thêm đường dẫn để python tìm thấy package stylus_tracking
from stylus_tracking.capture.video_capture import VideoCapture

def main():
    # CẤU HÌNH CAMERA:
    # - Webcam mặc định: video_source = 0
    # - IP Camera / DroidCam: ví dụ "http://172.16.1.249:4747/video"
    # video_source = "http://172.16.1.249:4747/video"  # Hãy đổi thành IP DroidCam của bạn hoặc số 0
    
    # print(f"Connecting to camera source: {video_source}...")
    try:
        # cap = VideoCapture(video_source=video_source, flip_horizontal=False)
        cap = VideoCapture()
    except Exception as e:
        print(f"Error: {e}")
        print("Please check if the camera source is active or if the IP is correct.")
        sys.exit(1)
        
    print("Camera connected successfully! Press 'q' or 'ESC' to quit.")
    
    while True:
        ret, frame = cap.get_next_frame()
        if not ret or frame is None:
            print("Could not grab frame from camera.")
            break
            
        # LƯU Ý QUAN TRỌNG:
        # Hàm cap.get_next_frame() trả về ảnh ở hệ màu RGB (để phục vụ hiển thị Tkinter/Pillow sau này).
        # Tuy nhiên, cv2.imshow của OpenCV mặc định hiển thị màu ở hệ BGR.
        # Do đó, ta cần chuyển đổi ngược từ RGB sang BGR để hiển thị đúng màu (không bị biến dạng mặt người thành màu xanh).
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Hiển thị ảnh trực tiếp lên cửa sổ
        cv2.imshow("Test Camera Feed (Press ESC to close)", frame_bgr)
        
        # Lắng nghe phím nhấn trong 1ms
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # 27 là mã ASCII của phím ESC
            break
            
    cv2.destroyAllWindows()
    print("Closed camera feed test successfully.")

if __name__ == "__main__":
    main()
