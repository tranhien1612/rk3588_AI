#!/usr/bin/env python3
import json
import os
import signal
import sys
import time
import urllib.request
import urllib.error
import cv2
import numpy as np
import inspireface as isf

from config import (
    MODEL_PACK_PATH,
    DATABASE_PATH,
    FRAME_URL,
    FACE_API_URL,
    SHM_FACE_CROP,
    SNAPSHOT_FILE,
    SIMILARITY_THRESHOLD,
    RECOGNIZE_INTERVAL,
    GENDER_MAP,
    AGE_BRACKET_MAP,
    EMOTION_MAP,
)
from database import FaceDatabase

g_running = True


def sig_handler(sig, frame):
    global g_running
    g_running = False


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


def post_face_result(data):
    """Ghi kết quả nhận diện khuôn mặt trực tiếp vào RAM SHM (< 0.05ms)"""
    try:
        body = json.dumps(data, ensure_ascii=False)
        with open("/dev/shm/face_tmp.json", "w", encoding="utf-8") as f:
            f.write(body)
        os.replace("/dev/shm/face_tmp.json", "/dev/shm/face_result.json")
    except Exception:
        pass


SHM_FRAME_PATH = "/dev/shm/frame_latest.jpg"

def fetch_latest_frame():
    """Lấy frame mới nhất: Ưu tiên đọc trực tiếp từ RAM (/dev/shm) < 0.5ms"""
    if os.path.exists(SHM_FRAME_PATH):
        try:
            img = cv2.imread(SHM_FRAME_PATH)
            if img is not None and img.size > 0:
                return img
        except Exception:
            pass
    try:
        with urllib.request.urlopen(FRAME_URL, timeout=0.2) as resp:
            data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    except Exception:
        return None


def main():
    print("=" * 70)
    print("FACE RECOGNITION REALTIME ENGINE (INSPIREFACE)")
    print("=" * 70)

    # 1. Khởi tạo Database
    db = FaceDatabase(DATABASE_PATH)
    person_count = len(db.persons)
    print(f"Database: {DATABASE_PATH}")
    print(f"Số lượng khuôn mặt đã đăng ký: {person_count}")
    if person_count == 0:
        print("CẢNH BÁO: Chưa có khuôn mặt nào trong database! Hãy chạy enroll.py để nạp dữ liệu.")

    # 2. Khởi tạo InspireFace SDK
    print(f"Đang tải model InspireFace: {MODEL_PACK_PATH}...")
    if not isf.launch(str(MODEL_PACK_PATH)):
        print("Lỗi: Không thể khởi chạy InspireFace SDK.")
        sys.exit(1)

    opt = isf.HF_ENABLE_FACE_RECOGNITION | isf.HF_ENABLE_FACE_ATTRIBUTE | isf.HF_ENABLE_FACE_EMOTION
    session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT, 10, -1, -1)
    print("InspireFace Engine đã khởi động thành công.")
    print(f"Đang đọc video từ: {FRAME_URL}")
    print(f"Đẩy kết quả tới: {FACE_API_URL}")
    print("Nhấn Ctrl+C để dừng.")
    print("-" * 70)

    last_had_face = False
    consecutive_no_face = 0

    while g_running:
        t_start = time.perf_counter()

        frame = fetch_latest_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]

        # Phát hiện khuôn mặt
        faces = session.face_detection(frame)

        if faces:
            consecutive_no_face = 0
            last_had_face = True

            # Sắp xếp các khuôn mặt theo kích thước (lớn nhất trước), giới hạn tối đa 5 mặt
            sorted_faces = sorted(
                faces,
                key=lambda f: (f.location[2] - f.location[0]) * (f.location[3] - f.location[1]),
                reverse=True
            )[:5]

            face_list = []
            seen_matched_names = set()

            for idx, face in enumerate(sorted_faces):
                x1, y1, x2, y2 = face.location

                # Trích xuất vector đặc trưng và nhận diện trong database
                feat = session.face_feature_extract(frame, face)
                match, sim = db.search(feat, SIMILARITY_THRESHOLD)

                if match:
                    person_name = match.get("name", "---")
                    # Chống trùng lặp: nếu cùng 1 người xuất hiện nhiều box thì chỉ giữ box lớn nhất
                    if person_name in seen_matched_names:
                        continue
                    seen_matched_names.add(person_name)
                    crop_id = len(seen_matched_names) - 1

                    # Crop khuôn mặt thực tế chỉ khi đã xác định danh tính
                    pad_x = int((x2 - x1) * 0.25)
                    pad_y = int((y2 - y1) * 0.30)
                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)
                    cx2 = min(w, x2 + pad_x)
                    cy2 = min(h, y2 + pad_y)

                    crop_img = frame[cy1:cy2, cx1:cx2]
                    if crop_img.size > 0:
                        try:
                            cv2.imwrite(f"/dev/shm/face_crop_{crop_id}.jpg", crop_img)
                            if crop_id == 0:
                                cv2.imwrite(str(SHM_FACE_CROP), crop_img)
                        except Exception:
                            pass

                    # Thuộc tính: Giới tính, độ tuổi, cảm xúc suy luận từ model InspireFace
                    ext = session.face_pipeline(frame, [face], opt)
                    gender_val = ext[0].gender if ext else -1
                    age_val = ext[0].age_bracket if ext else -1
                    emotion_val = ext[0].emotion if ext else -1

                    gender_str = GENDER_MAP.get(gender_val, "---")
                    age_str = AGE_BRACKET_MAP.get(age_val, "---")
                    emotion_str = EMOTION_MAP.get(emotion_val, "Bình thường")

                    face_data = {
                        "matched": True,
                        "crop_id": crop_id,
                        "name": person_name,
                        "age": f"{match.get('age')} tuổi" if match.get("age") else age_str,
                        "gender": gender_str,   # Lấy trực tiếp từ model suy luận
                        "emotion": emotion_str, # Cảm xúc từ model suy luận
                        "hometown": match.get("hometown", "---"),
                        "phone": match.get("phone", "---"),
                        "info": match.get("role", "---"),
                        "notes": match.get("notes", "---"),
                        "sim": round(sim, 3),
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                    }
                    # print(f"[NHẬN DIỆN #{crop_id+1}] {person_name} | Sim: {sim:.1%} | {gender_str} | {emotion_str}")
                else:
                    # Khuôn mặt không xác định: CHỈ gửi box để vẽ lên web, KHÔNG tạo thông tin card
                    face_data = {
                        "matched": False,
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                    }
                    # print(f"[KHÔNG XÁC ĐỊNH] Chỉ vẽ box khung hình (Sim: {sim:.1%})")

                face_list.append(face_data)

            payload = {
                "matched": any(f["matched"] for f in face_list),
                "faces": face_list,
                "ts": int(time.time() * 1000)
            }
            post_face_result(payload)

        else:
            consecutive_no_face += 1
            if last_had_face and consecutive_no_face >= 2:
                last_had_face = False
                post_face_result({"matched": False, "faces": [], "ts": int(time.time() * 1000)})

        dt = time.perf_counter() - t_start
        sleep_time = max(0.01, RECOGNIZE_INTERVAL - dt)
        time.sleep(sleep_time)

    # Dọn dẹp khi thoát
    post_face_result({"matched": False})
    session.release()
    db.close()
    print("Face recognition đã dừng.")


if __name__ == "__main__":
    main()
