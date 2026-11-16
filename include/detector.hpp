// Detector YOLO: pool N worker (mỗi worker 1 context RKNN, ghim NPU core), "frame mới nhất thắng".
#pragma once
#include <atomic>
#include <thread>
#include <vector>

#include "common.hpp"
#include "config.hpp"

namespace gv {

class Detector {
public:
    Detector(const Config& cfg, FrameMailbox& mb, Latest<DetResult>& out) : cfg_(cfg), mb_(mb), out_(out) {}
    ~Detector() { stop(); }
    bool start();   // false nếu không nạp được model
    void stop();
    FpsCounter fps;
    std::atomic<double> avg_infer_ms{0};

private:
    void worker(int idx);
    const Config& cfg_;
    FrameMailbox& mb_;
    Latest<DetResult>& out_;
    std::vector<std::thread> ths_;
    std::atomic<bool> run_{false};
    uint64_t claimed_ = 0;  // chỉ truy cập trong FrameMailbox::claim_latest (có khoá)
    uint64_t published_ = 0;  // bảo vệ bởi pub_m_
    std::mutex pub_m_;
    struct Impl;
    std::vector<Impl*> impls_;
};

}  // namespace gv
