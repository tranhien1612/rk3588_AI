#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand Gesture Gimbal Control Service (MediaPipe 21 Hand Landmarks) - High Performance
Điều khiển Gimbal thông minh bằng cử chỉ tay:
1. Xòe tay (Open Palm) giữ 2 giây: Bật/Tắt chế độ điều khiển cử chỉ.
2. Nắm tay (Fist): Tracking gimbal theo tâm bàn tay (Landmark 9).
Tối ưu hóa: Zero-copy RAM shared memory (/dev/shm) giảm thiểu tối đa độ trễ.
"""

import json
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import signal
import sys
import time
import urllib.request
import urllib.error
import cv2
import numpy as np
import mediapipe as mp

from config import (
    FRAME_URL,
    TRACK_HAND_API_URL,
    GESTURE_API_URL,
    SHM_FRAME_PATH,
    SHM_GESTURE_PATH,
    TRACK_HAND_TTL_MS,
    MODEL_COMPLEXITY,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)
from gesture_detector import HandGestureDetector

g_running = True

def sig_handler(sig, frame):
    global g_running
    g_running = False

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


import struct

def fetch_latest_frame():
    """Lấy frame mới nhất: Đọc trực tiếp từ RAM (/dev/shm) với IMREAD_REDUCED_COLOR_2 siêu tốc (< 3.5ms)"""
    if os.path.exists(SHM_FRAME_PATH):
        try:
            img = cv2.imread(SHM_FRAME_PATH, cv2.IMREAD_REDUCED_COLOR_2)
            if img is not None and img.size > 0:
                return img
        except Exception:
            pass

    # Fallback qua HTTP nếu chưa có file SHM
    try:
        with urllib.request.urlopen(FRAME_URL, timeout=0.15) as resp:
            data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_REDUCED_COLOR_2)
            return img
    except Exception:
        return None


def send_track_hand(x, y, ttl_ms=TRACK_HAND_TTL_MS):
    """Ghi trực tiếp tọa độ mục tiêu tâm bàn tay vào RAM SHM cho Gimbal PID (< 0.01ms)"""
    expire_t = time.monotonic() + (ttl_ms / 1000.0)
    data = struct.pack("ddd", float(x), float(y), float(expire_t))
    try:
        with open("/dev/shm/hand_target_tmp.bin", "wb") as f:
            f.write(data)
        os.replace("/dev/shm/hand_target_tmp.bin", "/dev/shm/hand_target.bin")
    except Exception:
        pass


def clear_track_hand():
    """Xóa mục tiêu bám tay ngay khi không còn nắm tay"""
    if os.path.exists("/dev/shm/hand_target.bin"):
        try:
            os.remove("/dev/shm/hand_target.bin")
        except Exception:
            pass


def send_gesture_result(result_dict):
    """Ghi kết quả nhận diện cử chỉ trực tiếp vào RAM SHM (< 0.05ms)"""
    try:
        body = json.dumps(result_dict, ensure_ascii=False)
        with open("/dev/shm/gesture_tmp.json", "w", encoding="utf-8") as f:
            f.write(body)
        os.replace("/dev/shm/gesture_tmp.json", "/dev/shm/gesture_result.json")
    except Exception:
        pass


def main():
    print("=" * 70)
    print("HAND GESTURE GIMBAL CONTROLLER (MediaPipe 21 Landmarks) - REALTIME")
    print("=" * 70)
    print("Đang khởi tạo MediaPipe Hands Engine...")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    detector = HandGestureDetector()

    print("MediaPipe Hands Engine đã sẵn sàng.")
    print(f"- Nguồn frame: {SHM_FRAME_PATH} (fallback {FRAME_URL})")
    print(f"- Cử chỉ 1 (Xòe tay): Giữ 2 giây để BẬT / TẮT")
    print(f"- Cử chỉ 2 (Nắm tay): Gimbal tự động tracking theo tâm bàn tay (Landmark 9)")
    print("Nhấn Ctrl+C để dừng.")
    print("-" * 70)

    fps_count = 0
    fps_time = time.time()
    last_frame_mtime = 0

    while g_running:
        t_loop_start = time.perf_counter()

        # Kiểm tra lệnh tắt chế độ cử chỉ từ phím Space trên web
        if os.path.exists("/dev/shm/gesture_off"):
            try:
                os.remove("/dev/shm/gesture_off")
            except Exception:
                pass
            detector.control_enabled = False
            detector.reset_hold()
            detector.last_toggle_time = time.time() + 1.5  # Cooldown 1.5s
            # print("[GESTURE] >>> ĐÃ TẮT CHẾ ĐỘ ĐIỀU KHIỂN CỬ CHỈ (Lệnh từ phím Space)")

        # Đồng bộ chặt chẽ với frame mới nhất (Zero-wait polling)
        try:
            curr_mtime = os.stat(SHM_FRAME_PATH).st_mtime_ns
        except Exception:
            curr_mtime = 0

        if curr_mtime == last_frame_mtime and curr_mtime > 0:
            time.sleep(0.002)
            continue
        last_frame_mtime = curr_mtime

        frame = fetch_latest_frame()
        if frame is None:
            time.sleep(0.005)
            continue

        # Resize nhỏ lại bằng phép nội suy NEAREST siêu tốc để tối ưu CPU ARM64
        small = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        mp_res = hands.process(rgb)

        if mp_res.multi_hand_landmarks:
            hand_lms = mp_res.multi_hand_landmarks[0]
            # Trích xuất 21 điểm landmarks chuẩn hóa [0, 1]
            raw_pts = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]

            res = detector.analyze_hand(raw_pts)
            # QUAN TRỌNG: CHỈ gửi landmarks khi chế độ điều khiển cử chỉ đang BẬT (ON)
            # Khi chế độ TẮT (OFF): Luôn để rỗng [] để tuyệt đối không vẽ điểm landmark nào lên web
            if res["mode"] == "ON":
                res["landmarks"] = [[round(p[0], 4), round(p[1], 4)] for p in raw_pts]
            else:
                res["landmarks"] = []
            res["ts"] = round(time.time(), 3)

            # Thực thi tracking khi Mode đang BẬT (ON) và đang NẮM TAY (FIST)
            if res["mode"] == "ON" and res["gesture"] == "FIST" and res["hand_center"]:
                hx, hy = res["hand_center"]
                send_track_hand(hx, hy, ttl_ms=TRACK_HAND_TTL_MS)
            else:
                clear_track_hand()

            send_gesture_result(res)

        else:
            # Không phát hiện bàn tay trong frame
            detector.reset_hold()
            clear_track_hand()

            res = {
                "mode": "ON" if detector.control_enabled else "OFF",
                "gesture": "NONE",
                "hold_sec": 0.0,
                "hold_target": 2.0,
                "progress": 0.0,
                "landmarks": [],
                "hand_center": None,
                "ts": round(time.time(), 3)
            }
            send_gesture_result(res)

        fps_count += 1
        if time.time() - fps_time >= 5.0:
            current_fps = fps_count / (time.time() - fps_time)
            fps_count = 0
            fps_time = time.time()



    print("Đang dừng dịch vụ cử chỉ...")
    hands.close()
    print("Dịch vụ đã dừng an toàn.")


if __name__ == "__main__":
    main()
