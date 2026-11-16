# Cài đặt (Orange Pi 5 Plus / RK3588, Ubuntu 20.04)

Yêu cầu đã có trên máy: g++ ≥ 9, GStreamer 1.16 (+ `mppvideodec`, `mppjpegenc`, `rtspsrc`), `librga` (`/usr/include/rga`), MPP.
Nếu thiếu:
```bash
sudo apt install -y build-essential pkg-config libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-rockchip1 librga-dev librockchip-mpp-dev
sudo usermod -aG dialout $USER     # quyền serial, đăng nhập lại
```
Không cần cmake/OpenCV. Build: `make -j8` → `build/gimbal_vision`.

**RKNN runtime**: dùng bản 1.5.2 kèm theo trong `third_party/rknn/librknnrt.so` (nạp qua rpath; thư viện hệ thống 1.4.0 không dùng vì thiếu API cần thiết). Kernel driver NPU cần tương thích (kiểm tra: `dmesg | grep -i rknpu`).

**Model** (`models/`): `yolov5s-640-640.rknn`, `FastSAM_S.rknn` (từ 2 repo mẫu), nhãn COCO. Model khác phải convert bằng rknn-toolkit2 trên PC; YOLO phải cùng định dạng 3 đầu ra int8 như repo mẫu.

Chạy nền bằng systemd (ví dụ `/etc/systemd/system/gimbal-vision.service`):
```
[Service]
WorkingDirectory=/home/orangepi/Desktop/web
ExecStart=/home/orangepi/Desktop/web/build/gimbal_vision
Restart=always
User=orangepi
```
