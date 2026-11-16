#!/bin/bash
# ======================================================================
# MASTER LAUNCHER - GIMBAL VISION + FACE RECOGNIZE + HAND GESTURE
# Khởi chạy toàn bộ hệ thống chỉ bằng 1 lệnh duy nhất
# ======================================================================

set -e
cd "$(dirname "$0")"

# 1. Biên dịch nhị phân C++ nếu chưa có hoặc có file source mới hơn
if [ ! -x build/gimbal_vision ]; then
    echo "[BUILD] Đang biên dịch gimbal_vision..."
    make -j"$(nproc)" || exit 1
fi

export LD_LIBRARY_PATH="$(pwd)/third_party/onnxruntime/lib:$(pwd)/third_party/rknn:$LD_LIBRARY_PATH"

# 2. Dọn dẹp tiến trình cũ (nếu có) và file rác trong RAM SHM
pkill -f "./build/gimbal_vision" 2>/dev/null || true
pkill -f "face_recognize/recognize.py" 2>/dev/null || true
pkill -f "hand_gesture/main.py" 2>/dev/null || true
rm -f /dev/shm/frame_*.jpg /dev/shm/hand_target* /dev/shm/gesture* /dev/shm/face* 2>/dev/null || true
sleep 0.5

# 3. Hàm dọn dẹp khi người dùng nhấn Ctrl+C hoặc tắt tiến trình
cleanup() {
    echo ""
    echo "======================================================================"
    echo "[SHUTDOWN] Đang dừng toàn bộ hệ thống Gimbal Vision..."
    echo "======================================================================"
    kill -TERM "$PID_VISION" "$PID_FACE" "$PID_GESTURE" 2>/dev/null || true
    wait "$PID_VISION" "$PID_FACE" "$PID_GESTURE" 2>/dev/null || true
    rm -f /dev/shm/frame_*.jpg /dev/shm/hand_target* /dev/shm/gesture* /dev/shm/face* 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM EXIT

# 4. Khởi chạy Gimbal Vision Native Core (C++ RKNN, GStreamer, MAVLink, WebServer)
CONFIG_FILE="${1:-config/config.json}"
echo "[1/3] Khởi chạy Gimbal Vision Core..."
./build/gimbal_vision "$CONFIG_FILE" &
PID_VISION=$!

# Đợi 1.5s để WebServer và luồng camera RTSP khởi tạo hoàn tất
sleep 1.5

# 5. Khởi chạy Face Recognize Service (InspireFace Realtime trên Big Core 7)
echo "[2/3] Khởi chạy Face Recognition Service..."
taskset -c 7 python3 face_recognize/recognize.py &
PID_FACE=$!

# 6. Khởi chạy Hand Gesture Service (MediaPipe 21 Landmarks trên 3 Big Cores 4,5,6)
echo "[3/3] Khởi chạy Hand Gesture Service..."
taskset -c 4,5,6 python3 hand_gesture/main.py &
PID_GESTURE=$!

echo ""
echo "======================================================================"
echo " HỆ THỐNG GIMBAL VISION TOÀN DIỆN ĐÃ HOẠT ĐỘNG"
echo " - Giao diện Web:     http://localhost:8080"
echo " - Gimbal Vision Core PID: $PID_VISION"
echo " - Nhận diện khuôn mặt PID: $PID_FACE"
echo " - Điều khiển cử chỉ   PID: $PID_GESTURE"
echo "======================================================================"
echo ""

# Chờ các tiến trình kết thúc
wait
