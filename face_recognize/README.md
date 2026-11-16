# Hệ Thống Nhận Diện Khuôn Mặt (InspireFace) - Gimbal Vision

Hệ thống nhận diện khuôn mặt thời gian thực sử dụng **InspireFace SDK**, trích xuất đặc trưng khuôn mặt 512 chiều, phân tích cảm xúc (*Vui vẻ, Bình thường, Ngạc nhiên, Buồn...*), giới tính, độ tuổi và tích hợp đồng bộ với giao diện Web Gimbal Vision.

---

## 1. Cấu trúc thư mục

```
web/face_recognize/
├── README.md           # Tài liệu hướng dẫn cài đặt & vận hành
├── requirements.txt    # Danh sách thư viện Python cần thiết
├── config.py           # Cấu hình đường dẫn, model, ngưỡng similarity, API kết nối
├── database.py         # Quản trị CSDL SQLite (faces.db) & thuật toán so khớp vector
├── enroll.py           # Script embedding dữ liệu gốc vào database
├── enroll.sh           # Lệnh chạy nhanh nạp dữ liệu: ./enroll.sh
├── recognize.py        # Script nhận diện realtime (đọc frame từ web, snapshot, đối chiếu CSDL)
├── run.sh              # Lệnh chạy nhanh nhận diện realtime: ./run.sh
└── data/
    ├── faces/          # Thư mục chứa các file ảnh khuôn mặt gốc (.jpg, .png)
    ├── persons.json    # File JSON mô tả thông tin chi tiết từng người
    ├── faces.db        # CSDL SQLite lưu thông tin và vector embedding 512 chiều
    └── last_snapshot.jpg # Ảnh snapshot khuôn mặt thực tế mới nhất
```

---

## 2. Hướng dẫn cài đặt môi trường & thư viện

Cài đặt các gói phụ thuộc qua pip:

```bash
cd /home/orangepi/Desktop/web/face_recognize
python3 -m pip install -r requirements.txt
```

Các thư viện chính bao gồm:
- `inspireface>=1.2.4`: SDK nhận diện khuôn mặt, phân tích cảm xúc & giới tính.
- `opencv-python-headless`: Xử lý cắt ảnh (snapshot) và định dạng khung hình.
- `numpy<2.0`: Xử lý ma trận và tính toán khoảng cách cosine similarity.

---

## 3. Lệnh embedding dữ liệu gốc vào Database

Dữ liệu gốc được lưu trữ tại `web/face_recognize/data/faces/`:
- **Quy tắc:** Mỗi file ảnh khuôn mặt đi kèm **1 file JSON cùng tên** mô tả thông tin của người đó.
  Ví dụ:
  - `Tran_Ba_Hien.jpg` đi kèm `Tran_Ba_Hien.json`
  - `person1.jpg` đi kèm `person1.json`

### Định dạng file JSON đi kèm ảnh (ví dụ `Tran_Ba_Hien.json`):
```json
{
  "name": "Trần Bá Hiến",
  "age": 26,
  "gender": "Nam",
  "hometown": "Bắc Ninh",
  "phone": "0988.123.456",
  "role": "Nhân viên IT",
  "notes": "Nhân sự VIP - Quyền truy cập cấp 2",
  "image": "Tran_Ba_Hien.jpg"
}
```

### Lệnh thực hiện embedding vào CSDL:
```bash
cd /home/orangepi/Desktop/web/face_recognize
./enroll.sh
# hoặc:
python3 enroll.py
```
> Script sẽ tự động quét tất cả các file ảnh trong `data/faces/`, đọc thông tin từ file `.json` cùng tên tương ứng, trích xuất vector đặc trưng 512 chiều với InspireFace và lưu vào file CSDL `data/faces.db`.

---

## 4. Lệnh chạy tính năng nhận diện khuôn mặt Realtime

Trước khi chạy, đảm bảo server chính Gimbal Vision đang hoạt động (`./run.sh` trong thư mục `web`).

Chạy tiến trình nhận diện:
```bash
cd /home/orangepi/Desktop/web/face_recognize
./run.sh
# hoặc:
python3 recognize.py
```

---

## 5. Luồng hoạt động chi tiết

```
[Camera RTSP] ──> [Gimbal Vision Server] ──> [/frame.jpg]
                           │                      │
                           ▼                      ▼
                  [Giao diện Web 8080]    [recognize.py]
                           ▲                      │
                           │                      ▼
                           │             Phát hiện khuôn mặt
                           │                      │
                           │                      ▼
                           │             Chụp snapshot khuôn mặt
                           │                      │
                           │                      ▼
                           │             Đối chiếu CSDL faces.db
                           │             & Phân tích cảm xúc, giới tính
                           │                      │
                           └────── POST /api/face_result ──────┘
```

1. **Phát hiện liên tục:** `recognize.py` liên tục kiểm tra khung hình video từ web.
2. **Snapshot tự động:** Khi phát hiện khuôn mặt, hệ thống ngay lập tức cắt ảnh thực tế của khuôn mặt đó (snapshot) lưu vào `/dev/shm/face_crop.jpg`.
3. **Đối chiếu & Phân tích:**
   - Trích xuất vector đặc trưng và đối chiếu với `faces.db` theo độ tương đồng Cosine Similarity.
   - Phân tích cảm xúc hiện tại (*Vui vẻ, Bình thường, Ngạc nhiên, Tức giận...*), giới tính và độ tuổi.
4. **Hiển thị trên Web:**
   - Góc dưới bên phải xuất hiện Card thông tin gồm: ảnh snapshot thực tế, Tên, Tuổi, Giới tính, Quê quán, SĐT, Vai trò, Cảm xúc.
   - Trên video hiển thị khung định vị khuôn mặt màu xanh ngọc nhạt.
   - **Tự động ẩn:** Khi khuôn mặt rời khỏi khung hình, Card thông tin lập tức biến mất để giữ giao diện sạch sẽ.
