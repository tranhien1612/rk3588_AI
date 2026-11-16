#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path
import cv2
import inspireface as isf

from config import (
    MODEL_PACK_PATH,
    FACE_DIR,
    DATABASE_PATH,
)
from database import FaceDatabase

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    print("=" * 70)
    print("EMBEDDING DỮ LIỆU KHUÔN MẶT VÀO DATABASE (INSPIREFACE)")
    print("Format: Mỗi file ảnh đi kèm 1 file JSON cùng tên.")
    print("=" * 70)

    if not MODEL_PACK_PATH.exists():
        print(f"Lỗi: Không tìm thấy model pack tại: {MODEL_PACK_PATH}")
        sys.exit(1)

    if not FACE_DIR.exists():
        print(f"Lỗi: Không tìm thấy thư mục ảnh: {FACE_DIR}")
        sys.exit(1)

    # Quét tất cả các file ảnh trong thư mục FACE_DIR
    image_files = sorted(
        [p for p in FACE_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )

    if not image_files:
        print(f"Không tìm thấy file ảnh nào trong: {FACE_DIR}")
        sys.exit(0)

    print(f"Đã tìm thấy {len(image_files)} file ảnh trong thư mục {FACE_DIR.name}/")

    # Khởi tạo InspireFace SDK
    print("Đang khởi tạo InspireFace SDK...")
    if not isf.launch(str(MODEL_PACK_PATH)):
        print("Lỗi: Không thể khởi chạy InspireFace.")
        sys.exit(1)

    opt = isf.HF_ENABLE_FACE_RECOGNITION | isf.HF_ENABLE_FACE_ATTRIBUTE | isf.HF_ENABLE_FACE_EMOTION
    session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT, 10, -1, -1)
    print("InspireFace session đã sẵn sàng.")

    # Khởi tạo Database
    db = FaceDatabase(DATABASE_PATH)

    success_count = 0
    fail_count = 0

    for idx, img_path in enumerate(image_files, start=1):
        key = img_path.stem
        json_path = img_path.with_suffix(".json")

        print("-" * 70)
        print(f"[{idx}/{len(image_files)}] Xử lý: {img_path.name}")

        info = {}
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                print(f"  [JSON] Đã đọc thông tin từ: {json_path.name}")
            except Exception as e:
                print(f"  [CẢNH BÁO] Lỗi đọc file JSON {json_path.name}: {e}")
        else:
            print(f"  [THÔNG BÁO] Không có file {json_path.name}, sử dụng thông tin mặc định.")

        # Đảm bảo có các trường thông tin cơ bản
        if "name" not in info or not info["name"]:
            info["name"] = key
        if "image" not in info:
            info["image"] = img_path.name

        print(f"  Thông tin: Tên='{info.get('name')}', Tuổi='{info.get('age')}', Quê='{info.get('hometown')}'")

        # Đọc ảnh
        img = cv2.imread(str(img_path))
        if img is None:
            print("  [LỖI] Không thể đọc file ảnh bằng OpenCV.")
            fail_count += 1
            continue

        t0 = time.perf_counter()
        faces = session.face_detection(img)
        dt = (time.perf_counter() - t0) * 1000

        if not faces:
            print(f"  [CẢNH BÁO] Không phát hiện khuôn mặt nào trong ảnh ({dt:.1f}ms).")
            fail_count += 1
            continue

        # Chọn khuôn mặt có diện tích lớn nhất trong ảnh
        face = max(faces, key=lambda f: (f.location[2] - f.location[0]) * (f.location[3] - f.location[1]))
        x1, y1, x2, y2 = face.location
        print(f"  Phát hiện khuôn mặt: tọa độ [{x1}, {y1}, {x2}, {y2}] ({dt:.1f}ms)")

        # Trích xuất vector đặc trưng
        feat = session.face_feature_extract(img, face)
        if feat is None:
            print("  [LỖI] Không thể trích xuất vector đặc trưng.")
            fail_count += 1
            continue

        # Lưu vào Database
        db.insert_person(key, info, feat)
        print(f"  [THÀNH CÔNG] Đã lưu embedding ({len(feat)} chiều) cho '{info.get('name')}'.")
        success_count += 1

    db.load_embeddings()
    db.close()
    session.release()

    print("=" * 70)
    print(f"KẾT QUẢ: Thành công {success_count}/{len(image_files)} | Thất bại: {fail_count}")
    print(f"Database đã được cập nhật: {DATABASE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
