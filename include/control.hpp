// Điều khiển gimbal: PID + DualAxisTrackerController + MAVLink (chuẩn xác 100% theo z1Camera_mavlink)
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <thread>
#include <utility>

#include "common.hpp"
#include "config.hpp"

namespace gv {

class PIDController {
public:
    double kp = 0.6;
    double ki = 0.0;
    double kd = 0.1;
    double max_out = 0.8;
    double min_out = -0.8;
    double integral_limit = 0.5;
    double integral_decay = 0.98;

    PIDController(double kp = 0.6, double ki = 0.0, double kd = 0.1,
                  double max_out = 0.8, double min_out = -0.8,
                  double integral_limit = 0.5, double integral_decay = 0.98)
        : kp(kp), ki(ki), kd(kd),
          max_out(max_out), min_out(min_out),
          integral_limit(integral_limit), integral_decay(integral_decay),
          integral(0.0), last_error(0.0), filtered_derivative(0.0),
          has_last_error(false), has_last_time(false) {}

    void reset() {
        integral = 0.0;
        last_error = 0.0;
        filtered_derivative = 0.0;
        has_last_error = false;
        has_last_time = false;
    }

    void reset_integral() {
        integral = 0.0;
    }

    double update(double error, double dt = -1.0) {
        auto now = std::chrono::steady_clock::now();
        if (dt <= 0.0) {
            if (!has_last_time) {
                dt = 0.033;
            } else {
                std::chrono::duration<double> elapsed = now - last_time;
                dt = elapsed.count();
            }
        }
        dt = std::clamp(dt, 0.01, 0.1);
        last_time = now;
        has_last_time = true;

        double p_term = kp * error;

        // Zero-crossing reset
        if (has_last_error && (error * last_error < 0.0)) {
            integral = 0.0;
        }

        integral = (integral * integral_decay) + (error * dt);
        if (integral_limit > 0.0) {
            integral = std::clamp(integral, -integral_limit, integral_limit);
        }
        double i_term = ki * integral;

        double raw_derivative = 0.0;
        if (has_last_error && dt > 0.0) {
            raw_derivative = (error - last_error) / dt;
            raw_derivative = std::clamp(raw_derivative, -4.0, 4.0);
        }

        if (!has_last_error) {
            filtered_derivative = 0.0;
        } else {
            filtered_derivative = 0.70 * filtered_derivative + 0.30 * raw_derivative;
        }

        double d_term = kd * filtered_derivative;
        last_error = error;
        has_last_error = true;

        double output = p_term + i_term + d_term;
        return std::clamp(output, min_out, max_out);
    }

private:
    double integral = 0.0;
    double last_error = 0.0;
    double filtered_derivative = 0.0;
    bool has_last_error = false;
    std::chrono::steady_clock::time_point last_time;
    bool has_last_time = false;
};

struct TrackingOutput {
    double yaw_rate = 0.0;
    double pitch_rate = 0.0;
};

class DualAxisTrackerController {
public:
    PIDController pid_yaw;
    PIDController pid_pitch;
    double deadband = 0.04;
    double max_rate_yaw = 0.8;
    double max_rate_pitch = 0.8;
    bool invert_yaw = true;
    bool invert_pitch = true;

    DualAxisTrackerController(
        double kp_yaw, double ki_yaw, double kd_yaw,
        double kp_pitch, double ki_pitch, double kd_pitch,
        double deadband,
        double max_rate_yaw, double max_rate_pitch,
        bool invert_yaw, bool invert_pitch
    ) : pid_yaw(kp_yaw, ki_yaw, kd_yaw, max_rate_yaw, -max_rate_yaw),
        pid_pitch(kp_pitch, ki_pitch, kd_pitch, max_rate_pitch, -max_rate_pitch),
        deadband(deadband),
        max_rate_yaw(max_rate_yaw), max_rate_pitch(max_rate_pitch),
        invert_yaw(invert_yaw), invert_pitch(invert_pitch) {}

    void reset() {
        pid_yaw.reset();
        pid_pitch.reset();
        last_cmd_yaw_ = 0.0;
        last_cmd_pitch_ = 0.0;
        has_last_step_ = false;
    }

    TrackingOutput update(
        const std::pair<double, double>& target_center,
        const std::pair<int, int>& frame_size
    ) {
        double fw = (frame_size.first > 0) ? frame_size.first : 1920.0;
        double fh = (frame_size.second > 0) ? frame_size.second : 1080.0;
        double img_cx = fw / 2.0;
        double img_cy = fh / 2.0;

        double diff_x = target_center.first - img_cx;
        double diff_y = target_center.second - img_cy;

        double effective_deadband = deadband * (fw / 2.0);
        double error_dist = std::hypot(diff_x, diff_y);

        if (error_dist <= effective_deadband) {
            pid_yaw.reset_integral();
            pid_pitch.reset_integral();
            last_cmd_yaw_ = 0.0;
            last_cmd_pitch_ = 0.0;
            return {0.0, 0.0};
        }

        double norm_error_x = diff_x / (fw / 2.0);
        double norm_error_y = diff_y / (fh / 2.0);

        pid_yaw.max_out = max_rate_yaw;
        pid_yaw.min_out = -max_rate_yaw;
        double raw_yaw = pid_yaw.update(norm_error_x);

        pid_pitch.max_out = max_rate_pitch;
        pid_pitch.min_out = -max_rate_pitch;
        double raw_pitch = -pid_pitch.update(norm_error_y);

        double cmd_yaw = invert_yaw ? -raw_yaw : raw_yaw;
        double cmd_pitch = invert_pitch ? -raw_pitch : raw_pitch;

        cmd_yaw = std::clamp(cmd_yaw, -max_rate_yaw, max_rate_yaw);
        cmd_pitch = std::clamp(cmd_pitch, -max_rate_pitch, max_rate_pitch);

        auto now = std::chrono::steady_clock::now();
        double dt = 0.033;
        if (has_last_step_) {
            std::chrono::duration<double> el = now - last_step_time_;
            dt = std::clamp(el.count(), 0.01, 0.1);
        }
        last_step_time_ = now;
        has_last_step_ = true;

        // Slew rate limiting (Max angular acceleration: 2.5 rad/s^2)
        double max_accel = 2.5; // rad/s^2
        double max_step = max_accel * dt;

        double limited_yaw = std::clamp(cmd_yaw, last_cmd_yaw_ - max_step, last_cmd_yaw_ + max_step);
        double limited_pitch = std::clamp(cmd_pitch, last_cmd_pitch_ - max_step, last_cmd_pitch_ + max_step);

        last_cmd_yaw_ = limited_yaw;
        last_cmd_pitch_ = limited_pitch;

        return {limited_yaw, limited_pitch};
    }

private:
    double last_cmd_yaw_ = 0.0;
    double last_cmd_pitch_ = 0.0;
    std::chrono::steady_clock::time_point last_step_time_;
    bool has_last_step_ = false;
};

class MavlinkLink {
public:
    explicit MavlinkLink(const Config& c) : cfg_(c) {}
    ~MavlinkLink() { stop(); }
    bool start();
    void stop();
    bool connected() const { return connected_; }
    void send_rate(double yaw_rate, double pitch_rate);
    void sync_target_to_current();
    void recenter();
    double yaw_deg() const { return cur_yaw_; }
    double pitch_deg() const { return cur_pitch_; }

private:
    bool open_port();
    void io_loop();
    bool write_msg(const void* msg);
    const Config& cfg_;
    int fd_ = -1;
    std::atomic<bool> run_{false}, connected_{false}, has_att_{false};
    std::thread th_;
    std::mutex wm_, cm_;
    std::atomic<double> cur_yaw_{0}, cur_pitch_{0};
    double tgt_yaw_ = 0, tgt_pitch_ = 0, last_cmd_t_ = 0;
    bool has_last_cmd_time_ = false;
};

class GimbalController {
public:
    GimbalController(const Config& c, Latest<TrackResult>& trk, MavlinkLink& mav)
        : cfg_(c), trk_(trk), mav_(mav),
          ctrl_(c.kp_yaw, c.ki_yaw, c.kd_yaw,
                c.kp_pitch, c.ki_pitch, c.kd_pitch,
                c.deadband, c.max_rate, c.max_rate,
                c.invert_yaw, c.invert_pitch) {}
    ~GimbalController() { stop(); }
    void start();
    void stop();
    void reset();
    void set_manual(double yaw_rate, double pitch_rate, int ttl_ms);
    void set_hand_target(double norm_x, double norm_y, int ttl_ms);
    std::atomic<double> out_yaw{0}, out_pitch{0};
    std::atomic<bool> tracking_active{false};
    std::atomic<bool> hand_tracking_active{false};

private:
    void run();
    const Config& cfg_;
    Latest<TrackResult>& trk_;
    MavlinkLink& mav_;
    DualAxisTrackerController ctrl_;
    std::thread th_;
    std::atomic<bool> run_{false};
    std::mutex m_;
    double man_yaw_ = 0, man_pitch_ = 0, man_until_ = 0;
    double hand_x_ = -1, hand_y_ = -1, hand_until_ = 0;
};

}  // namespace gv

