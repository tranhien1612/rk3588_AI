#include <pthread.h>
#include <sched.h>
#include <signal.h>

#include <cstdio>
#include <cstring>

#include "capture.hpp"
#include "config.hpp"
#include "control.hpp"
#include "detector.hpp"
#include "tracker.hpp"
#include "webserver.hpp"

using namespace gv;

static std::atomic<bool> g_run{true};
static void on_sig(int) { g_run = false; }

static void pin_self(const std::vector<int>& cpus) {
    if (cpus.empty()) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    for (int c : cpus) CPU_SET(c, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

int main(int argc, char** argv) {
    std::string cfg_path = argc > 1 ? argv[1] : "config/config.json";
    Config cfg;
    try {
        cfg = Config::load(cfg_path);
    } catch (const std::exception& e) {
        fprintf(stderr, "LỖI CẤU HÌNH: %s\n", e.what());
        return 2;
    }
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    signal(SIGPIPE, SIG_IGN);

    FrameMailbox mb;
    Latest<JpegFrame> jpeg;
    Latest<DetResult> det_res;
    Latest<TrackResult> trk_res;

    // I/O threads (capture, web, mavlink) -> core LITTLE
    pin_self(cfg.aff_io);
    MavlinkLink mav(cfg);
    mav.start();
    Capture cap(cfg, mb, jpeg);
    cap.start();

    // AI threads -> core big
    pin_self(cfg.aff_ai);
    Detector det(cfg, mb, det_res);
    bool det_ok = false;
    if (cfg.det_enabled) {
        det_ok = det.start();
        if (!det_ok) fprintf(stderr, "CẢNH BÁO: detector không chạy (kiểm tra model/NPU).\n");
    }

    TrackerManager trk(cfg, mb, trk_res);
    trk.start();

    pin_self(cfg.aff_io);
    GimbalController ctl(cfg, trk_res, mav);
    ctl.start();

    AppState st;
    st.cfg = &cfg; st.cap = &cap; st.det = &det; st.trk = &trk; st.mav = &mav; st.ctl = &ctl;
    st.jpeg = &jpeg; st.det_res = &det_res; st.trk_res = &trk_res;
    WebServer web(st);
    if (!web.start()) { g_run = false; }

    double last_log = now_s();
    while (g_run) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        cap.fps.update(); det.fps.update(); trk.fps.update();
        if (now_s() - last_log > 5.0) {
            last_log = now_s();
            fprintf(stderr, "[stat] rtsp=%d cap=%.1f det=%.1f (%.1fms) trk=%.1f mav=%d\n",
                    (int)cap.connected(), cap.fps.fps, det.fps.fps, det.avg_infer_ms.load(),
                    trk.fps.fps, (int)mav.connected());
        }
    }

    fprintf(stderr, "Đang dừng hệ thống...\n");
    web.stop();
    ctl.stop();
    mav.send_rate(0, 0);
    cap.stop();
    trk.stop();
    det.stop();
    mav.stop();
    return 0;
}
