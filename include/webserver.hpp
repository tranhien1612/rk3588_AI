// Web server tối giản (socket thuần, không phụ thuộc): trang web, MJPEG, SSE overlay, API
#pragma once
#include <atomic>
#include <thread>

#include "capture.hpp"
#include "common.hpp"
#include "config.hpp"
#include "control.hpp"
#include "detector.hpp"
#include "tracker.hpp"

namespace gv {

struct AppState {
    const Config* cfg = nullptr;
    Capture* cap = nullptr;
    Detector* det = nullptr;
    TrackerManager* trk = nullptr;
    MavlinkLink* mav = nullptr;
    GimbalController* ctl = nullptr;
    Latest<JpegFrame>* jpeg = nullptr;
    Latest<DetResult>* det_res = nullptr;
    Latest<TrackResult>* trk_res = nullptr;
};

class WebServer {
public:
    WebServer(AppState s) : s_(s) {}
    ~WebServer() { stop(); }
    bool start();
    void stop();

private:
    void accept_loop();
    void handle(int fd);
    void serve_mjpeg(int fd);
    void serve_events(int fd);
    std::string overlay_json();
    AppState s_;
    int lfd_ = -1;
    std::atomic<bool> run_{false};
    std::atomic<int> clients_{0};
    std::thread th_;
    std::mutex face_m_;
    std::string latest_face_json_;
    double latest_face_t_ = 0;
    std::mutex gesture_m_;
    std::string latest_gesture_json_;
    double latest_gesture_t_ = 0;
};

}  // namespace gv

