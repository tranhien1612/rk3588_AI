# Cấu hình Hand Gesture Gimbal Controller
import os
from pathlib import Path

# URL dịch vụ WebServer & Bộ nhớ chia sẻ RAM (SHM)
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
FRAME_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/frame.jpg"
TRACK_HAND_API_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/api/track_hand"
MANUAL_API_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/api/manual"
GESTURE_API_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/api/gesture_result"

# Tối ưu hóa Realtime: Đọc/Ghi trực tiếp RAM SHM (Zero-socket, Zero-latency)
SHM_FRAME_PATH = "/dev/shm/frame_latest.jpg"
SHM_GESTURE_PATH = "/dev/shm/gesture_result.json"

# MediaPipe cấu hình
MODEL_COMPLEXITY = 0          # 0: Lite (rất nhanh, tối ưu CPU ARM64), 1: Full
MIN_DETECTION_CONFIDENCE = 0.55
MIN_TRACKING_CONFIDENCE = 0.50
FRAME_WIDTH = 320              # Tối ưu kích thước đầu vào MediaPipe CPU ARM64 siêu tốc (18ms, 55 FPS)
FRAME_HEIGHT = 180

# 1. Cử chỉ Xòe tay (Open Palm) - Giảm xuống 2 giây theo yêu cầu
OPEN_PALM_HOLD_SECONDS = 2.0  # Thời gian giữ xòe tay liên tục để bật/tắt (2s)
TOGGLE_COOLDOWN_SECONDS = 1.5 # Thời gian chờ sau khi bật/tắt trước khi nhận lần tiếp

# 2. Cử chỉ Nắm tay (Fist) - Điều khiển Gimbal theo tâm bàn tay (Landmark 9)
TRACK_HAND_TTL_MS = 180       # Thời gian hiệu lực mục tiêu bàn tay mỗi frame gửi
