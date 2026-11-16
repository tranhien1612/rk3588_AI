#include "capture.hpp"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <cstdio>
#include <cstring>

namespace gv {

static bool is_rtsp(const std::string& u) { return u.rfind("rtsp://", 0) == 0 || u.rfind("rtsps://", 0) == 0; }

std::string Capture::pipeline() const {
    std::string src;
    if (is_rtsp(cfg_.rtsp_url)) {
        std::string depay = cfg_.codec == "h265" ? "rtph265depay ! h265parse" : "rtph264depay ! h264parse";
        src = "rtspsrc location=\"" + cfg_.rtsp_url + "\" latency=" + std::to_string(cfg_.latency_ms) +
              " protocols=" + cfg_.transport + " drop-on-latency=true buffer-mode=none ! " + depay +
              " ! mppvideodec ! video/x-raw,format=NV12";
    } else {
        // nguồn file (chỉ để test)
        src = "filesrc location=\"" + cfg_.rtsp_url + "\" ! decodebin ! videoconvert ! video/x-raw,format=NV12";
        if (cfg_.file_realtime) src += " ! identity sync=true";
    }
    const char* q = "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0";
    const char* sink = "appsink drop=true max-buffers=1 sync=false";
    std::string p = src + " ! tee name=t t. ! " + q + " ! " + sink + " name=ai t. ! " + q +
                    " ! mppjpegenc quant=" + std::to_string(cfg_.jpeg_quant) + " ! " + sink + " name=jpg";
    return p;
}

void Capture::start() {
    run_ = true;
    th_ = std::thread([this] { run(); });
}

void Capture::stop() {
    run_ = false;
    mb_.stop();
    if (th_.joinable()) th_.join();
}

void Capture::run() {
    gst_init(nullptr, nullptr);
    uint64_t frame_id = 0, jseq = 0;
    while (run_) {
        std::string desc = pipeline();
        fprintf(stderr, "[capture] pipeline: %s\n", desc.c_str());
        GError* err = nullptr;
        GstElement* pipe = gst_parse_launch(desc.c_str(), &err);
        if (!pipe) {
            fprintf(stderr, "[capture] parse lỗi: %s\n", err ? err->message : "?");
            if (err) g_error_free(err);
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.reconnect_ms));
            continue;
        }
        GstElement* ai = gst_bin_get_by_name(GST_BIN(pipe), "ai");
        GstElement* jp = gst_bin_get_by_name(GST_BIN(pipe), "jpg");
        gst_element_set_state(pipe, GST_STATE_PLAYING);

        // luồng JPEG
        std::atomic<bool> jrun{true};
        std::thread jth([&] {
            while (jrun) {
                GstSample* s = gst_app_sink_try_pull_sample(GST_APP_SINK(jp), 200 * GST_MSECOND);
                if (!s) continue;
                GstBuffer* b = gst_sample_get_buffer(s);
                GstMapInfo m;
                if (b && gst_buffer_map(b, &m, GST_MAP_READ)) {
                    auto j = std::make_shared<JpegFrame>();
                    j->seq = ++jseq;
                    j->t = now_s();
                    j->data.assign(m.data, m.data + m.size);
                    jpeg_.set(j);

                    // Ghi frame trực tiếp vào RAM /dev/shm cho các tiến trình AI đọc siêu tốc
                    FILE* fp = fopen("/dev/shm/frame_tmp.jpg", "wb");
                    if (fp) {
                        fwrite(m.data, 1, m.size, fp);
                        fclose(fp);
                        rename("/dev/shm/frame_tmp.jpg", "/dev/shm/frame_latest.jpg");
                    }

                    gst_buffer_unmap(b, &m);
                }
                gst_sample_unref(s);
            }
        });

        // Pre-allocated Frame pool (4 frames vòng tròn, triệt tiêu 100% chi phí malloc/free 93MB/s)
        std::vector<std::shared_ptr<Frame>> pool;
        for (int pi = 0; pi < 4; pi++) {
            auto pf = std::make_shared<Frame>();
            pf->nv12.reserve(1920 * 1088 * 3 / 2);
            pool.push_back(pf);
        }
        size_t pool_idx = 0;

        GstBus* bus = gst_element_get_bus(pipe);
        uint64_t last_rx = 0;
        double last_ok = now_s();
        while (run_) {
            GstSample* s = gst_app_sink_try_pull_sample(GST_APP_SINK(ai), 200 * GST_MSECOND);
            if (!s) {
                GstMessage* msg = gst_bus_pop_filtered(bus, (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
                if (msg) {
                    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                        GError* e = nullptr; gst_message_parse_error(msg, &e, nullptr);
                        fprintf(stderr, "[capture] lỗi pipeline: %s\n", e ? e->message : "?");
                        if (e) g_error_free(e);
                    } else fprintf(stderr, "[capture] EOS\n");
                    gst_message_unref(msg);
                    break;
                }
                if (now_s() - last_ok > 3.0) { fprintf(stderr, "[capture] mất frame >3s, kết nối lại\n"); break; }
                continue;
            }
            GstCaps* caps = gst_sample_get_caps(s);
            GstVideoInfo vi;
            GstBuffer* b = gst_sample_get_buffer(s);
            if (caps && b && gst_video_info_from_caps(&vi, caps)) {
                GstVideoFrame vf;
                if (gst_video_frame_map(&vf, &vi, b, GST_MAP_READ)) {
                    int w = GST_VIDEO_FRAME_WIDTH(&vf), h = GST_VIDEO_FRAME_HEIGHT(&vf);
                    auto f = pool[pool_idx++ % pool.size()];
                    f->w = w; f->h = h;
                    int st = (w + 15) & ~15;
                    f->stride = st;
                    size_t req_sz = (size_t)st * h * 3 / 2;
                    if (f->nv12.size() != req_sz) f->nv12.resize(req_sz);
                    const uint8_t* ys = (const uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vf, 0);
                    int ysd = GST_VIDEO_FRAME_PLANE_STRIDE(&vf, 0);
                    const uint8_t* uvs = (const uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vf, 1);
                    int uvsd = GST_VIDEO_FRAME_PLANE_STRIDE(&vf, 1);
                    uint8_t* yd = f->nv12.data();
                    uint8_t* uvd = yd + (size_t)st * h;
                    if (w == st && ysd == st && uvsd == st) {
                        memcpy(yd, ys, (size_t)st * h);
                        memcpy(uvd, uvs, (size_t)st * (h / 2));
                    } else {
                        for (int r = 0; r < h; r++) memcpy(yd + (size_t)r * st, ys + (size_t)r * ysd, w);
                        for (int r = 0; r < h / 2; r++) memcpy(uvd + (size_t)r * st, uvs + (size_t)r * uvsd, w);
                    }
                    f->id = ++frame_id;
                    f->t_capture = now_s();
                    w_ = w; h_ = h;
                    mb_.put(f);
                    fps.tick();
                    connected_ = true;
                    last_ok = f->t_capture;
                    last_rx++;
                    gst_video_frame_unmap(&vf);
                }
            }
            gst_sample_unref(s);
        }
        connected_ = false;
        jrun = false;
        gst_element_set_state(pipe, GST_STATE_NULL);
        if (jth.joinable()) jth.join();
        gst_object_unref(bus);
        gst_object_unref(ai);
        gst_object_unref(jp);
        gst_object_unref(pipe);
        if (run_) std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.reconnect_ms));
    }
}

}  // namespace gv

