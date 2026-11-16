#include "control.hpp"

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "mavlink/common/mavlink.h"

namespace gv {

// ---------------- MAVLink ----------------
static speed_t baud_of(int b) {
    switch (b) {
        case 9600: return B9600; case 19200: return B19200; case 38400: return B38400;
        case 57600: return B57600; case 115200: return B115200; case 230400: return B230400;
        case 460800: return B460800; case 921600: return B921600;
        default: return B115200;
    }
}

bool MavlinkLink::open_port() {
    fd_ = open(cfg_.mav_port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd_ < 0) return false;
    struct termios tty;
    if (tcgetattr(fd_, &tty) != 0) { close(fd_); fd_ = -1; return false; }
    cfsetospeed(&tty, baud_of(cfg_.mav_baud));
    cfsetispeed(&tty, baud_of(cfg_.mav_baud));
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~(IGNBRK | IXON | IXOFF | IXANY);
    tty.c_lflag = 0; tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0; tty.c_cc[VTIME] = 0;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD | CSTOPB | CRTSCTS);
    if (tcsetattr(fd_, TCSANOW, &tty) != 0) { close(fd_); fd_ = -1; return false; }
    return true;
}

bool MavlinkLink::start() {
    if (!cfg_.mav_enabled) return false;
    run_ = true;
    bool ok = open_port();
    connected_ = ok;
    fprintf(stderr, ok ? "[mavlink] mở %s @%d\n" : "[mavlink] chưa mở được %s (sẽ thử lại)\n", cfg_.mav_port.c_str(), cfg_.mav_baud);
    th_ = std::thread([this] { io_loop(); });
    return ok;
}

void MavlinkLink::stop() {
    if (run_) { send_rate(0, 0); }   // dừng gimbal an toàn
    run_ = false;
    if (th_.joinable()) th_.join();
    if (fd_ >= 0) { close(fd_); fd_ = -1; }
    connected_ = false;
}

bool MavlinkLink::write_msg(const void* m) {
    if (!connected_ || fd_ < 0) return false;
    uint8_t buf[MAVLINK_MAX_PACKET_LEN];
    uint16_t n = mavlink_msg_to_send_buffer(buf, (const mavlink_message_t*)m);
    std::lock_guard<std::mutex> l(wm_);
    ssize_t w = write(fd_, buf, n);
    if (w != (ssize_t)n) { connected_ = false; return false; }
    return true;
}

void MavlinkLink::io_loop() {
    double last_hb = 0, last_state = 0, last_try = now_s();
    mavlink_message_t msg;
    mavlink_status_t status;
    uint8_t rx[512];
    while (run_) {
        double t = now_s();
        if (!connected_) {   // tự mở lại serial
            if (t - last_try > 1.0) {
                last_try = t;
                if (fd_ >= 0) { close(fd_); fd_ = -1; }
                if (open_port()) { connected_ = true; fprintf(stderr, "[mavlink] đã kết nối lại %s\n", cfg_.mav_port.c_str()); }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        if (t - last_hb >= 0.5) {
            last_hb = t;
            mavlink_message_t hb;
            mavlink_msg_heartbeat_pack(cfg_.mav_sys, cfg_.mav_comp, &hb, MAV_TYPE_QUADROTOR, MAV_AUTOPILOT_ARDUPILOTMEGA,
                                       MAV_MODE_FLAG_SAFETY_ARMED, 0, MAV_STATE_ACTIVE);
            write_msg(&hb);
        }
        if (t - last_state >= 0.033) {   // GCU Z1-Mini yêu cầu luồng autopilot-state ~30Hz
            last_state = t;
            mavlink_message_t sm;
            float q[4] = {1, 0, 0, 0};
            uint64_t us = (uint64_t)(t * 1e6);
            mavlink_msg_autopilot_state_for_gimbal_device_pack(cfg_.mav_sys, cfg_.mav_comp, &sm, cfg_.mav_target_sys,
                                                               cfg_.mav_target_comp, us, q, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            write_msg(&sm);
        }
        struct pollfd p{fd_, POLLIN, 0};
        int r = poll(&p, 1, 5);
        if (r > 0 && (p.revents & POLLIN)) {
            ssize_t n = read(fd_, rx, sizeof(rx));
            if (n < 0) { connected_ = false; continue; }
            for (ssize_t i = 0; i < n; i++) {
                if (!mavlink_parse_char(MAVLINK_COMM_0, rx[i], &msg, &status)) continue;
                if (msg.msgid == MAVLINK_MSG_ID_COMMAND_LONG) {
                    mavlink_command_long_t c;
                    mavlink_msg_command_long_decode(&msg, &c);
                    if (c.command == MAV_CMD_SET_MESSAGE_INTERVAL) {
                        mavlink_message_t ack;
                        mavlink_msg_command_ack_pack(cfg_.mav_sys, cfg_.mav_comp, &ack, c.command, MAV_RESULT_ACCEPTED, 0, 0,
                                                     msg.sysid, msg.compid);
                        write_msg(&ack);
                    }
                } else if (msg.msgid == MAVLINK_MSG_ID_COMMAND_ACK) {
                    mavlink_command_ack_t ack;
                    mavlink_msg_command_ack_decode(&msg, &ack);
                    // fprintf(stderr, "[mavlink-ack] cmd=%u result=%u from sys=%u comp=%u\n",
                    //         ack.command, ack.result, msg.sysid, msg.compid);
                } else if (msg.msgid == MAVLINK_MSG_ID_GIMBAL_DEVICE_ATTITUDE_STATUS) {
                    mavlink_gimbal_device_attitude_status_t a;
                    mavlink_msg_gimbal_device_attitude_status_decode(&msg, &a);
                    float q0 = a.q[0], q1 = a.q[1], q2 = a.q[2], q3 = a.q[3];
                    cur_pitch_ = std::asin(std::clamp(2.0f * (q0 * q2 - q3 * q1), -1.0f, 1.0f)) * 180.0 / M_PI;
                    cur_yaw_ = std::atan2(2.0f * (q0 * q3 + q1 * q2), 1.0f - 2.0f * (q2 * q2 + q3 * q3)) * 180.0 / M_PI;
                    if (!has_att_) {
                        std::lock_guard<std::mutex> l(cm_);
                        tgt_pitch_ = cur_pitch_;
                        tgt_yaw_ = cur_yaw_;
                        has_att_ = true;
                    }
                } else if (msg.msgid == MAVLINK_MSG_ID_ATTITUDE) {
                    mavlink_attitude_t a;
                    mavlink_msg_attitude_decode(&msg, &a);
                    cur_pitch_ = a.pitch * 180.0 / M_PI;
                    cur_yaw_ = a.yaw * 180.0 / M_PI;
                    if (!has_att_) {
                        std::lock_guard<std::mutex> l(cm_);
                        tgt_pitch_ = cur_pitch_;
                        tgt_yaw_ = cur_yaw_;
                        has_att_ = true;
                    }
                }
            }
        }
    }
}

void MavlinkLink::sync_target_to_current() {
    std::lock_guard<std::mutex> l(cm_);
    if (has_att_) {
        tgt_yaw_ = cur_yaw_;
        tgt_pitch_ = cur_pitch_;
    }
    has_last_cmd_time_ = false;
}

void MavlinkLink::send_rate(double yaw_rate, double pitch_rate) {
    if (!connected_) return;
    std::lock_guard<std::mutex> l(cm_);

    // Zero-rate command: lock current physical orientation, do not integrate (chuẩn z1Camera_mavlink)
    if (std::abs(pitch_rate) < 1e-4 && std::abs(yaw_rate) < 1e-4) {
        if (has_att_) {
            tgt_yaw_ = cur_yaw_;
            tgt_pitch_ = cur_pitch_;
        } else {
            // Chưa có attitude thực tế từ GCU: không gửi lệnh đè (0, 0)
            return;
        }
    } else {
        double t = now_s();
        double dt = 0.033;
        if (has_last_cmd_time_) {
            dt = std::clamp(t - last_cmd_t_, 0.005, 0.08);
        }
        last_cmd_t_ = t;
        has_last_cmd_time_ = true;

        tgt_yaw_ += (yaw_rate * 180.0 / M_PI) * dt;
        tgt_pitch_ += (pitch_rate * 180.0 / M_PI) * dt;
        if (has_att_) {
            tgt_yaw_ = std::clamp(tgt_yaw_, cur_yaw_ - 5.0, cur_yaw_ + 5.0);
            tgt_pitch_ = std::clamp(tgt_pitch_, cur_pitch_ - 5.0, cur_pitch_ + 5.0);
        }
    }
    tgt_yaw_ = std::clamp(tgt_yaw_, -cfg_.max_yaw_deg, cfg_.max_yaw_deg);
    tgt_pitch_ = std::clamp(tgt_pitch_, cfg_.min_pitch_deg, cfg_.max_pitch_deg);

    double pr = tgt_pitch_ * M_PI / 180.0, yr = tgt_yaw_ * M_PI / 180.0;
    double cy = cos(yr * .5), sy = sin(yr * .5), cp = cos(pr * .5), sp = sin(pr * .5);
    float q[4] = {(float)(cp * cy), (float)(-sp * sy), (float)(sp * cy), (float)(cp * sy)};

    mavlink_message_t m1, m2, m3;
    mavlink_msg_command_long_pack(cfg_.mav_sys, cfg_.mav_comp, &m1, cfg_.mav_target_sys, cfg_.mav_target_comp,
                                  MAV_CMD_DO_MOUNT_CONTROL, 0, (float)tgt_pitch_, 0.f, (float)tgt_yaw_, 0.f, 0.f, 0.f,
                                  MAV_MOUNT_MODE_MAVLINK_TARGETING);
    write_msg(&m1);
    mavlink_msg_gimbal_device_set_attitude_pack(cfg_.mav_sys, cfg_.mav_comp, &m2, cfg_.mav_target_sys, cfg_.mav_target_comp,
                                                0, q, 0.f, (float)pitch_rate, (float)yaw_rate);
    write_msg(&m2);
    mavlink_msg_gimbal_manager_set_pitchyaw_pack(cfg_.mav_sys, cfg_.mav_comp, &m3, cfg_.mav_target_sys, cfg_.mav_target_comp,
                                                 0, 0, (float)pr, (float)yr, (float)pitch_rate, (float)yaw_rate);
    write_msg(&m3);
}

void MavlinkLink::recenter() {
    if (!connected_) return;
    { std::lock_guard<std::mutex> l(cm_); tgt_yaw_ = 0; tgt_pitch_ = 0; last_cmd_t_ = 0; has_last_cmd_time_ = false; }
    mavlink_message_t m1, m2;
    mavlink_msg_command_long_pack(cfg_.mav_sys, cfg_.mav_comp, &m1, cfg_.mav_target_sys, cfg_.mav_target_comp,
                                  MAV_CMD_DO_MOUNT_CONTROL, 0, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MAV_MOUNT_MODE_MAVLINK_TARGETING);
    write_msg(&m1);
    float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    mavlink_msg_gimbal_device_set_attitude_pack(cfg_.mav_sys, cfg_.mav_comp, &m2, cfg_.mav_target_sys, cfg_.mav_target_comp,
                                                GIMBAL_DEVICE_FLAGS_NEUTRAL, q, 0.0f, 0.0f, 0.0f);
    write_msg(&m2);
}

// ---------------- GimbalController ----------------
void GimbalController::start() {
    run_ = true;
    th_ = std::thread([this] { run(); });
}

void GimbalController::stop() {
    run_ = false;
    if (th_.joinable()) th_.join();
}

void GimbalController::reset() {
    std::lock_guard<std::mutex> l(m_);
    ctrl_.reset();
    out_yaw = 0;
    out_pitch = 0;
    hand_x_ = -1;
    hand_y_ = -1;
    hand_until_ = 0;
    hand_tracking_active = false;
    unlink("/dev/shm/hand_target.bin");
}

void GimbalController::set_manual(double y, double p, int ttl_ms) {
    std::lock_guard<std::mutex> l(m_);
    man_yaw_ = y; man_pitch_ = p; man_until_ = now_s() + ttl_ms / 1000.0;
}

void GimbalController::set_hand_target(double norm_x, double norm_y, int ttl_ms) {
    std::lock_guard<std::mutex> l(m_);
    hand_x_ = norm_x;
    hand_y_ = norm_y;
    hand_until_ = now_s() + ttl_ms / 1000.0;
}

void GimbalController::run() {
    const double period = 1.0 / cfg_.ctl_hz;
    double next = now_s();
    bool was_active_cmd = false;

    while (run_) {
        next += period;
        double t = now_s();
        auto tr = trk_.get();
        double cmd_y = 0.0, cmd_p = 0.0;
        bool active = false;

        // 1. Kiểm tra mục tiêu Hand Tracking:
        // Đọc trực tiếp từ RAM SHM (/dev/shm/hand_target.bin) siêu tốc < 0.001ms
        double h_x = -1, h_y = -1;
        {
            int hfd = open("/dev/shm/hand_target.bin", O_RDONLY);
            if (hfd >= 0) {
                double target_data[3]; // x, y, expire_t (monotonic time)
                if (read(hfd, target_data, sizeof(target_data)) == sizeof(target_data)) {
                    if (t < target_data[2]) {
                        h_x = target_data[0];
                        h_y = target_data[1];
                    }
                }
                close(hfd);
            }
        }
        if (h_x < 0) {
            std::lock_guard<std::mutex> l(m_);
            if (t < hand_until_) {
                h_x = hand_x_;
                h_y = hand_y_;
            }
        }

        if (h_x >= 0 && h_y >= 0) {
            // Chạy thuật toán PID tracker bám theo tâm bàn tay (Landmark 9)
            std::pair<double, double> target_center = {h_x * 1920.0, h_y * 1080.0};
            std::pair<int, int> frame_sz = {1920, 1080};
            TrackingOutput rates = ctrl_.update(target_center, frame_sz);
            cmd_y = rates.yaw_rate;
            cmd_p = rates.pitch_rate;
            active = true;
            hand_tracking_active = true;
        } else if (tr && tr->valid() && tr->w > 0 && (t - tr->t_capture) * 1000 < cfg_.lost_timeout_ms) {
            hand_tracking_active = false;
            std::pair<double, double> target_center = {tr->cx, tr->cy};
            std::pair<int, int> frame_sz = {tr->w, tr->h};

            TrackingOutput rates = ctrl_.update(target_center, frame_sz);
            cmd_y = rates.yaw_rate;
            cmd_p = rates.pitch_rate;
            active = true;
        } else {
            hand_tracking_active = false;
            std::lock_guard<std::mutex> l(m_);
            if (t < man_until_) {
                cmd_y = man_yaw_;
                cmd_p = man_pitch_;
                active = true;
            }
        }

        tracking_active = active;
        out_yaw = cmd_y;
        out_pitch = cmd_p;

        // CHỈ gửi lệnh MAVLink khi đang có hành động điều khiển (tracking hoặc manual)
        // Khi ở trạng thái IDLE (không điều khiển): Tuyệt đối KHÔNG gửi lệnh đè lên Gimbal
        if (active) {
            mav_.send_rate(cmd_y, cmd_p);
            was_active_cmd = true;
        } else if (was_active_cmd) {
            // Khi vừa dừng điều khiển: gửi lệnh dừng (rate=0) một lần duy nhất và khóa góc
            ctrl_.reset();
            mav_.send_rate(0.0, 0.0);
            mav_.sync_target_to_current();
            was_active_cmd = false;
        }

        double sl = next - now_s();
        if (sl > 0) std::this_thread::sleep_for(std::chrono::duration<double>(sl));
        else next = now_s();
    }
}

}  // namespace gv

