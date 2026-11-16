// Capture: RTSP/file -> giải mã phần cứng MPP -> NV12 (AI) + JPEG phần cứng (web)
#pragma once
#include <atomic>
#include <thread>

#include "common.hpp"
#include "config.hpp"

namespace gv {

class Capture {
public:
    Capture(const Config& cfg, FrameMailbox& mailbox, Latest<JpegFrame>& jpeg)
        : cfg_(cfg), mb_(mailbox), jpeg_(jpeg) {}
    ~Capture() { stop(); }
    void start();
    void stop();
    bool connected() const { return connected_; }
    int width() const { return w_; }
    int height() const { return h_; }
    FpsCounter fps;

private:
    void run();
    std::string pipeline() const;
    const Config& cfg_;
    FrameMailbox& mb_;
    Latest<JpegFrame>& jpeg_;
    std::thread th_;
    std::atomic<bool> run_{false}, connected_{false};
    std::atomic<int> w_{0}, h_{0};
};

}  // namespace gv

