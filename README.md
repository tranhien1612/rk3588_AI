# Gimbal Vision (C++ / RK3588)

RTSP → giải mã MPP → **YOLO (RKNN, 3 NPU core)** luôn chạy · click trên web → **FastSAM** → **tracker + Kalman** → **PID** → **MAVLink** gimbal · hiển thị web (MJPEG phần cứng + overlay SSE).
Kế hoạch & kiến trúc: [PLAN.md](PLAN.md).

## Trạng thái triển khai

| Hạng mục | Trạng thái |
|---|---|
| Capture RTSP (GStreamer + `mppvideodec`, reconnect, nguồn file để test) | ✅ chạy, test bằng file |
| Detection YOLOv5s RKNN pool 3 core, "frame mới nhất thắng" | ✅ chạy: ~22 ms/frame/worker (3 worker ⇒ năng lực ~130 FPS; thực tế = FPS camera) |
| FastSAM-s click → bbox (mask chỉ tính cho instance chứa điểm click) | ✅ chạy, đã test click→track |
| Kalman + One-Euro + máy trạng thái (IDLE/SELECTING/TRACKING/LOW_CONF/LOST) | ✅ |
| **NanoTrack** | ⚠️ **chưa**: đang dùng `TemplateTracker` (ZNCC, CPU) qua giao diện `ITrackerBackend`. Cần model NanoTrack `.rknn` (convert từ ONNX trong `z1Camera_mavlink/models` bằng rknn-toolkit2 trên PC) để thay thế |
| PID + MAVLink (port từ `z1Camera_mavlink`) | ✅ biên dịch & logic chạy; **chưa thử với gimbal thật** (không có serial/camera trên máy lúc test) |
| Web: video + box detection + box tracking + click/Cancel/Recenter | ✅ (chưa mở bằng trình duyệt thật; đã test API/MJPEG/SSE bằng curl) |
| MediaPipe cử chỉ | ⏳ chỉ có hook `GimbalController::set_manual()` |
| Chưa làm: H264 passthrough/WebRTC, zero-copy dma-buf, tái định vị bằng YOLO | ⏳ |

Khác với PLAN: overlay dùng **SSE** (không dùng WebSocket) và web server tự viết bằng socket để không cần thư viện ngoài; FastSAM-s trong repo mẫu có input **640×640** (không phải 640×352).

## Cài đặt
Xem [docs/INSTALL.md](docs/INSTALL.md). Tóm tắt: `make -j8` (không cần cmake/OpenCV; RKNN runtime 1.5.2 + mavlink + json đã nằm trong `third_party/`).

## Chạy
```bash
sudo tools/perf_mode.sh     # (tuỳ chọn) cố định xung
./run.sh                    # mở http://<ip-board>:8080
```
Cấu hình duy nhất: `config/config.json` (đọc 1 lần lúc khởi động, sai sẽ báo lỗi và thoát).

## Sử dụng
- Trang web hiển thị video, box detection (kèm `class_id:tên score`), box tracking (đỏ).
- **Click** vào vật → FastSAM khoanh vùng (tô đỏ) → tracking → gimbal tự bám.
- `C` / nút **Hủy track**; `R` / nút **Recenter**.
- HUD: trạng thái RTSP/MAVLink, FPS, độ trễ, trạng thái track, góc gimbal.

## Test không cần camera
```bash
./build/gimbal_vision tools/test/config_test.json   # nguồn = tools/test/test.mp4, MAVLink tắt, cổng 8081
```
