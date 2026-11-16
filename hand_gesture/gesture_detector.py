import time
import math
from config import (
    OPEN_PALM_HOLD_SECONDS,
    TOGGLE_COOLDOWN_SECONDS,
)

# MediaPipe Hand Landmark IDs:
# 0: Wrist
# 1: Thumb CMC, 2: Thumb MCP, 3: Thumb IP, 4: Thumb TIP
# 5: Index MCP, 6: Index PIP, 7: Index DIP, 8: Index TIP
# 9: Middle MCP, 10: Middle PIP, 11: Middle DIP, 12: Middle TIP
# 13: Ring MCP, 14: Ring PIP, 15: Ring DIP, 16: Ring TIP
# 17: Pinky MCP, 18: Pinky PIP, 19: Pinky DIP, 20: Pinky TIP

def euclidean_dist_2d(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


class HandGestureDetector:
    def __init__(self):
        # Trạng thái bật/tắt chế độ điều khiển cử chỉ
        self.control_enabled = False

        # Bộ đếm giữ tay xòe (Open Palm)
        self.open_palm_start_time = None
        self.open_palm_held_sec = 0.0
        self.open_palm_progress = 0.0
        self.last_toggle_time = 0.0
        self.toggled_in_this_hold = False

    def reset_hold(self):
        self.open_palm_start_time = None
        self.open_palm_held_sec = 0.0
        self.open_palm_progress = 0.0
        self.toggled_in_this_hold = False

    def analyze_hand(self, lm):
        """
        lm: list gồm 21 tuple (x, y, z) chuẩn hóa [0, 1]
        Trả về:
            current_gesture: "OPEN_PALM" | "FIST" | "NONE"
            metadata: dict các thông số phân tích
        """
        now = time.time()

        # 1. Kiểm tra biên: Tuyệt đối không nhận diện bàn tay bị cắt xén ở mép khung hình
        for p in lm:
            if p[0] < 0.05 or p[0] > 0.95 or p[1] < 0.05 or p[1] > 0.95:
                self.reset_hold()
                return self._build_result("NONE", None, None)

        wrist = lm[0]
        middle_mcp = lm[9]

        # Kích thước bàn tay cơ sở (khoảng cách cổ tay đến gốc ngón giữa)
        palm_size = euclidean_dist_2d(wrist, middle_mcp)
        # Bàn tay phải có kích thước hợp lệ trong khung hình (không quá xa hoặc quá gần)
        if palm_size < 0.07 or palm_size > 0.45:
            self.reset_hold()
            return self._build_result("NONE", None, None)

        # Trục bàn tay từ Cổ tay (0) đến Gốc ngón giữa (9)
        axis_x = (middle_mcp[0] - wrist[0]) / palm_size
        axis_y = (middle_mcp[1] - wrist[1]) / palm_size

        # Bàn tay phải hướng lên về phía camera (wrist phải nằm dưới khớp gốc ngón tay)
        # Ngăn chặn việc bàn tay để trên bàn / gõ phím / chuột bị nhận nhầm
        if axis_y > -0.20:
            self.reset_hold()
            return self._build_result("NONE", None, None)

        # Định nghĩa 4 ngón dài: (mcp, pip, dip, tip)
        fingers = [
            (5, 6, 7, 8),     # Ngón Trỏ
            (9, 10, 11, 12),  # Ngón Giữa
            (13, 14, 15, 16), # Ngón Áp út
            (17, 18, 19, 20)  # Ngón Út
        ]

        open_fingers_count = 0
        closed_fingers_count = 0

        for mcp_idx, pip_idx, dip_idx, tip_idx in fingers:
            mcp = lm[mcp_idx]
            pip = lm[pip_idx]
            tip = lm[tip_idx]

            d_tip_wrist = euclidean_dist_2d(tip, wrist)
            d_pip_wrist = euclidean_dist_2d(pip, wrist)
            d_tip_mcp = euclidean_dist_2d(tip, mcp)

            # Vector từ mcp tới tip chiếu lên trục bàn tay
            v_tip = (tip[0] - mcp[0], tip[1] - mcp[1])
            proj_tip = dot_product(v_tip, (axis_x, axis_y))

            # Ngón duỗi thẳng: tip xa cổ tay hơn pip rõ rệt, vươn dài theo trục
            if d_tip_wrist > d_pip_wrist * 1.25 and d_tip_mcp > 0.70 * palm_size and proj_tip > 0.35 * palm_size:
                open_fingers_count += 1
            # Ngón gập: tip co sát vào mcp hoặc cổ tay
            elif d_tip_wrist <= d_pip_wrist * 1.05 and d_tip_mcp < 0.52 * palm_size:
                closed_fingers_count += 1

        # Ngón cái (Thumb: 1, 2, 3, 4)
        thumb_tip = lm[4]
        thumb_mcp = lm[2]
        d_thumb_tip_wrist = euclidean_dist_2d(thumb_tip, wrist)
        d_thumb_mcp_wrist = euclidean_dist_2d(thumb_mcp, wrist)
        d_thumb_index_mcp = euclidean_dist_2d(thumb_tip, lm[5])

        thumb_open = (d_thumb_index_mcp > 0.50 * palm_size and d_thumb_tip_wrist > d_thumb_mcp_wrist * 1.15)
        thumb_closed = (d_thumb_index_mcp < 0.58 * palm_size or euclidean_dist_2d(thumb_tip, middle_mcp) < 0.65 * palm_size)

        # 1. CỬ CHỈ XÒE TAY (Open Palm):
        # Yêu cầu: Cả 4 ngón dài mở + ngón cái mở + khoảng cách giữa các ngón xòe rộng
        spread_open = (euclidean_dist_2d(lm[8], lm[20]) > 0.65 * palm_size and
                       euclidean_dist_2d(lm[8], lm[12]) > 0.15 * palm_size)
        is_open_palm = (open_fingers_count == 4 and thumb_open and spread_open)

        # 2. CỬ CHỈ NẮM TAY (Fist):
        # Yêu cầu: Cả 4 ngón dài gập chặt + ngón cái gập + không phải xòe tay
        is_fist = (closed_fingers_count == 4 and thumb_closed and not is_open_palm)

        current_gesture = "NONE"
        if is_open_palm:
            current_gesture = "OPEN_PALM"
        elif is_fist:
            current_gesture = "FIST"

        # --- XỬ LÝ CỬ CHỈ 1: XÒE TAY (Open Palm) Giữ 2 giây để Bật/Tắt ---
        if is_open_palm:
            if self.open_palm_start_time is None:
                self.open_palm_start_time = now

            self.open_palm_held_sec = now - self.open_palm_start_time
            self.open_palm_progress = min(1.0, self.open_palm_held_sec / OPEN_PALM_HOLD_SECONDS)

            # Đủ 2 giây và hết thời gian chờ cooldown
            if (self.open_palm_held_sec >= OPEN_PALM_HOLD_SECONDS and 
                not self.toggled_in_this_hold and 
                (now - self.last_toggle_time) > TOGGLE_COOLDOWN_SECONDS):
                
                self.control_enabled = not self.control_enabled
                self.toggled_in_this_hold = True
                self.last_toggle_time = now
                # print(f"[GESTURE] >>> ĐÃ {'BẬT' if self.control_enabled else 'TẮT'} CHẾ ĐỘ ĐIỀU KHIỂN GESTURE (Giữ xòe tay 2s)")
        else:
            self.reset_hold()

        # Khi NẮM TAY (Fist): Lấy tọa độ Landmark 9 (Middle MCP) làm tâm bàn tay
        hand_center = None
        if is_fist:
            hand_center = [round(lm[9][0], 4), round(lm[9][1], 4)]

        return self._build_result(current_gesture, hand_center, now)

    def _build_result(self, current_gesture, hand_center, ts):
        return {
            "mode": "ON" if self.control_enabled else "OFF",
            "gesture": current_gesture,
            "hold_sec": round(self.open_palm_held_sec, 2),
            "hold_target": OPEN_PALM_HOLD_SECONDS,
            "progress": round(self.open_palm_progress, 3),
            "hand_center": hand_center,
            "ts": round(ts if ts else time.time(), 3)
        }
