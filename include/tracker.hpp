#pragma once
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"
#include "config.hpp"

namespace gv {

class NanoTrackBackend;

class TrackerManager {
public:
    TrackerManager(const Config& cfg, FrameMailbox& mb, Latest<TrackResult>& out);
    ~TrackerManager();

    void start();
    void stop();

    // Người dùng click kéo thả vẽ box tracking trên web (toạ độ chuẩn hoá 0..1)
    void select_box(float x1, float y1, float x2, float y2);
    void cancel();

    FpsCounter fps;

private:
    void run();

    const Config& cfg_;
    FrameMailbox& mb_;
    Latest<TrackResult>& out_;
    std::thread th_;
    std::atomic<bool> run_{false};

    std::mutex m_;
    bool has_sel_box_ = false;
    bool do_cancel_ = false;
    float sel_x1_ = 0, sel_y1_ = 0, sel_x2_ = 0, sel_y2_ = 0;

    std::unique_ptr<NanoTrackBackend> nano_;
};

}  // namespace gv
