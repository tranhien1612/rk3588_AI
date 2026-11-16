// Kiểu dữ liệu & tiện ích dùng chung
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gv {

inline double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Frame NV12 (đã bỏ stride), kích thước w*h*3/2
struct Frame {
    uint64_t id = 0;
    double t_capture = 0;
    int w = 0, h = 0;
    int stride = 0;                 // bước dòng (căn 16 cho RGA); NV12: Y h dòng + UV h/2 dòng
    std::vector<uint8_t> nv12;
    const uint8_t* y() const { return nv12.data(); }
};
using FramePtr = std::shared_ptr<const Frame>;

struct Box { float x1 = 0, y1 = 0, x2 = 0, y2 = 0; };  // pixel khung hình gốc

struct Det {
    Box box;
    float score = 0;
    int cls = 0;  // "id object" = class id
    std::string name;
};

struct DetResult {
    uint64_t frame_id = 0;
    double t_capture = 0, t_done = 0;
    int w = 0, h = 0;
    std::vector<Det> dets;
};

enum class TrackState { IDLE, SELECTING, TRACKING, LOW_CONF, LOST };
inline const char* to_str(TrackState s) {
    switch (s) {
        case TrackState::IDLE: return "IDLE";
        case TrackState::SELECTING: return "SELECTING";
        case TrackState::TRACKING: return "TRACKING";
        case TrackState::LOW_CONF: return "LOW_CONF";
        default: return "LOST";
    }
}

struct TrackResult {
    TrackState state = TrackState::IDLE;
    uint64_t frame_id = 0;
    double t_capture = 0;
    int w = 0, h = 0;
    Box box;            // đã lọc Kalman + làm mượt
    float cx = 0, cy = 0, vx = 0, vy = 0;  // px, px/s
    float score = 0;
    bool valid() const { return state == TrackState::TRACKING || state == TrackState::LOW_CONF; }
};

// Mailbox "frame mới nhất thắng": producer ghi đè, consumer lấy frame mới hơn id đã thấy.
class FrameMailbox {
public:
    void put(FramePtr f) {
        {
            std::lock_guard<std::mutex> l(m_);
            latest_ = std::move(f);
        }
        cv_.notify_all();
    }
    FramePtr latest() {
        std::lock_guard<std::mutex> l(m_);
        return latest_;
    }
    // Chờ frame có id > last_id; trả nullptr nếu hết thời gian / stop.
    FramePtr wait_newer(uint64_t last_id, int timeout_ms) {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait_for(l, std::chrono::milliseconds(timeout_ms),
                     [&] { return stop_ || (latest_ && latest_->id > last_id); });
        if (latest_ && latest_->id > last_id) return latest_;
        return nullptr;
    }
    // Dùng cho pool nhiều worker: lấy frame mới nhất chưa ai nhận (claim nguyên tử).
    FramePtr claim_latest(uint64_t& claimed_id, int timeout_ms) {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait_for(l, std::chrono::milliseconds(timeout_ms),
                     [&] { return stop_ || (latest_ && latest_->id > claimed_id); });
        if (latest_ && latest_->id > claimed_id) {
            claimed_id = latest_->id;
            return latest_;
        }
        return nullptr;
    }
    void stop() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; }
        cv_.notify_all();
    }
private:
    std::mutex m_;
    std::condition_variable cv_;
    FramePtr latest_;
    bool stop_ = false;
};

// Kho kết quả mới nhất (detection / tracking / jpeg)
template <class T>
class Latest {
public:
    void set(std::shared_ptr<const T> v) { std::lock_guard<std::mutex> l(m_); v_ = std::move(v); }
    std::shared_ptr<const T> get() { std::lock_guard<std::mutex> l(m_); return v_; }
private:
    std::mutex m_;
    std::shared_ptr<const T> v_;
};

// Bộ đếm FPS đơn giản
struct FpsCounter {
    std::atomic<uint64_t> n{0};
    uint64_t last_n = 0; double last_t = now_s(); double fps = 0;
    void tick() { n++; }
    double update() {
        double t = now_s();
        if (t - last_t >= 1.0) { uint64_t c = n; fps = (c - last_n) / (t - last_t); last_n = c; last_t = t; }
        return fps;
    }
};

struct JpegFrame { uint64_t seq = 0; double t = 0; std::vector<uint8_t> data; };

}  // namespace gv

