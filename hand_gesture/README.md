# Điều Khiển Gimbal Bằng Cử Chỉ Bàn Tay (MediaPipe Hand Landmarks)

Tính năng điều khiển Gimbal thông minh sử dụng **MediaPipe 21 Hand Landmarks**, nhận diện cử chỉ bàn tay từ video camera thời gian thực để tương tác và ra lệnh cho Gimbal.

---

## 1. Các Cử Chỉ & Hành Động

| Cử chỉ | Hành động | Mô tả chi tiết |
|---|---|---|
| **XÒE TAY (Open Palm)** | **Bật / Tắt chế độ điều khiển** | Giữ xòe cả 5 ngón tay liên tục trong **2 giây**. Khi đạt 2 giây hệ thống sẽ chuyển đổi trạng thái BẬT hoặc TẮT để tránh kích hoạt nhầm. |
| **NẮM TAY (Fist)** | **Gimbal bám theo bàn tay (Hand Tracking)** | Khi nắm tay, hệ thống lấy tọa độ tâm bàn tay (**Landmark 9 - gốc ngón giữa**) làm mục tiêu và kích hoạt bộ điều khiển PID Gimbal tự động xoay bám theo chuyển động của bàn tay nắm. |

> **Lưu ý:**
> - Khi chế độ cử chỉ đang **BẬT (ON)**: Hệ thống tự động **khóa tính năng kéo thả vẽ box tracking thủ công bằng chuột trên Web** để tránh xung đột thao tác.
> - Khi nắm tay điều khiển, hệ thống bám trực tiếp theo tâm bàn tay, **không hiển thị box object tracking rườm rà**.
> - Trên Web: **Chỉ hiển thị 21 điểm xương bàn tay** và các đường nối màu sắc mượt mà, không vẽ các vector hay ký hiệu che khuất video.

---

## 2. Cài Đặt Thư Viện

Module chạy trên môi trường Python 3 của Orange Pi / RK3588:

```bash
cd /home/orangepi/Desktop/web/hand_gesture
pip install mediapipe opencv-python numpy
```

---

## 3. Khởi Chạy Dịch Vụ

Dịch vụ chạy nền tự động lấy frame từ WebServer (`/frame.jpg`) và gửi kết quả tracking về Gimbal Controller:

```bash
./run.sh
```
