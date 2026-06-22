import cv2
import numpy as np
import os

# ==========================================
# CẤU HÌNH THÔNG SỐ (CONFIG)
# ==========================================
# Sử dụng bàn cờ có 10x7 góc trong (tương ứng 11x8 ô vuông) giống trong calibration.py
BOARD_SIZE = (10, 7)
# Kích thước thực tế của 1 ô vuông (mm) - mặc định 23mm
SQUARE_SIZE = 23.0 
# Tên file lưu trữ thông số hiệu chuẩn
SAVE_FILE = os.path.join(os.path.dirname(__file__), "camera_calibration_data.npz")

# Tiêu chí dừng thuật toán tinh chỉnh góc ô cờ (sub-pixel accuracy)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Chuẩn bị sẵn tọa độ 3D thực tế lý tưởng của các góc ô cờ (X, Y, Z) với Z = 0
objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Mảng lưu trữ điểm 3D (thực tế) và điểm 2D (trên ảnh) từ các frame thu thập được
objpoints = [] # Điểm 3D trong không gian thực
imgpoints = [] # Điểm 2D phát hiện trên ảnh chụp

def main():
    # Mở webcam (camera mặc định là 0, độ phân giải HD 720p)
    # Ví dụ địa chỉ hiển thị trên điện thoại là 192.168.42.129
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Lỗi: Không thể mở Webcam!")
        return

    print("=================================================================")
    print(" CHƯƠNG TRÌNH HIỆU CHUẨN CAMERA (STANDALONE CAMERA CALIBRATION)")
    print("=================================================================")
    print(" Hướng dẫn sử dụng:")
    print("  - Đưa tấm checkerboard (bàn cờ) phẳng trước camera.")
    print("  - Nhấn phím 'c' để chụp ảnh calib (cần tối thiểu 10 ảnh ở nhiều góc nghiêng).")
    print("  - Nhấn phím 'g' để bắt đầu tính toán ma trận camera và hệ số méo.")
    print("  - Nhấn phím 'q' để thoát chương trình.")
    print("=================================================================")

    calibrated = False
    cameraMatrix = None
    distCoef = None

    # Thử tải thông số calib cũ nếu đã có sẵn
    if os.path.exists(SAVE_FILE):
        try:
            data = np.load(SAVE_FILE)
            cameraMatrix = data['cameraMatrix']
            distCoef = data['distCoef']
            calibrated = True
            print(f" Đã tải thông số hiệu chuẩn có sẵn từ: {SAVE_FILE}")
        except Exception as e:
            print(" Không thể tải file cấu hình cũ, tiến hành thu thập mới.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không nhận được khung hình từ webcam.")
            break

        h, w = frame.shape[:2]
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Tìm các góc của bàn cờ
        ret_corners, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

        # Nếu tìm thấy bàn cờ, vẽ các góc lên màn hình trực quan
        if ret_corners:
            cv2.drawChessboardCorners(display_frame, BOARD_SIZE, corners, ret_corners)

        # 2. Xử lý trạng thái hiển thị
        status_text = f"Frames Captured: {len(objpoints)}/10+"
        if calibrated:
            status_text += " | Status: CALIBRATED (Active)"
        else:
            status_text += " | Status: UNCALIBRATED (Need chess board)"

        cv2.putText(display_frame, status_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if calibrated else (0, 0, 255), 2)
        cv2.putText(display_frame, "Keys: [c]=Capture frame | [g]=Run Calibrate | [q]=Quit", 
                    (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Hiển thị cửa sổ tương tác chính
        cv2.imshow("Webcam Feed & Chessboard Detection", display_frame)

        # 3. Nếu đã calib thành công, hiển thị thêm cửa sổ so sánh méo ảnh (Undistortion)
        if calibrated and cameraMatrix is not None and distCoef is not None:
            # Tính ma trận camera mới tối ưu hơn để không bị cắt rìa ảnh sau khi khử méo
            newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoef, (w, h), 1, (w, h))
            # Khử méo ảnh (Undistort)
            undistorted_frame = cv2.undistort(frame, cameraMatrix, distCoef, None, newCameraMatrix)
            
            # Cắt ảnh theo ROI nếu cần, hoặc hiển thị toàn bộ
            x, y, w_roi, h_roi = roi
            # Vẽ đường phân cách hoặc ghép đôi ảnh để xem side-by-side
            comparison = np.hstack((frame, undistorted_frame))
            # Resize để vừa màn hình máy tính
            comparison_resized = cv2.resize(comparison, (w, h // 2))
            
            # Hiển thị cửa sổ so sánh ảnh gốc và ảnh sau khi sửa méo
            cv2.putText(comparison_resized, "RAW DISTORTED (GOC)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(comparison_resized, "UNDISTORTED (KHU MEO)", (w // 2 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow("Undistort Comparison (Side-by-Side)", comparison_resized)

        # 4. Xử lý các sự kiện nhấn phím
        key = cv2.waitKey(1) & 0xFF
        
        # Thoát chương trình
        if key == ord('q'):
            break

        # Chụp ảnh góc ô cờ
        elif key == ord('c'):
            if ret_corners:
                # Làm mịn góc ô cờ tăng độ chính xác
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                print(f"[OK] Đã lưu thành công Frame thứ {len(objpoints)}")
                
                # Hiệu ứng chớp màn hình trắng khi lưu ảnh thành công
                flash = np.ones_like(frame) * 255
                cv2.imshow("Webcam Feed & Chessboard Detection", flash)
                cv2.waitKey(100)
            else:
                print("[LỖI] Không tìm thấy tấm bàn cờ trong khung hình hiện tại! Hãy căn góc lại.")

        # Tính toán hiệu chuẩn camera (chạy hiệu chuẩn khi có tối thiểu 10 ảnh)
        elif key == ord('g'):
            if len(objpoints) < 10:
                print(f"[CẢNH BÁO] Cần tối thiểu 10 ảnh để calib chính xác. Hiện tại mới có {len(objpoints)} ảnh.")
                print("Nếu vẫn muốn ép buộc tính toán, hãy chụp thêm vài ảnh nữa.")
                continue

            print("\nĐang tiến hành hiệu chuẩn camera... Vui lòng đợi trong giây lát...")
            # Chạy thuật toán chính của OpenCV
            ret_val, cameraMatrix, distCoef, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            if ret_val:
                print("\n==============================================")
                print(" HIỆU CHUẨN CAMERA THÀNH CÔNG!")
                print("==============================================")
                print(f" Sai số RMS (Re-projection Error): {ret_val:.4f} pixels")
                print("\n Ma trận nội tham Camera Matrix (3x3):")
                print(cameraMatrix)
                print("\n Hệ số méo thấu kính Dist Coef (5 tham số):")
                print(distCoef.ravel())
                print("==============================================")

                # Lưu thông số xuống file npz
                np.savez(SAVE_FILE, cameraMatrix=cameraMatrix, distCoef=distCoef)
                print(f" Đã lưu cấu hình thành công vào: {SAVE_FILE}\n")
                calibrated = True
            else:
                print("[LỖI] Thuật toán hiệu chuẩn thất bại! Vui lòng chụp lại ảnh.")

    cap.release()
    cv2.destroyAllWindows()
    print("Chương trình kết thúc.")

if __name__ == "__main__":
    main()
