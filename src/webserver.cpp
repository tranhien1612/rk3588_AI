#include "webserver.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "nlohmann/json.hpp"

namespace gv {
using nlohmann::json;

static bool send_all(int fd, const void* d, size_t n) {
    const char* p = (const char*)d;
    while (n) {
        ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) return false;
        p += w; n -= w;
    }
    return true;
}
static bool send_str(int fd, const std::string& s) { return send_all(fd, s.data(), s.size()); }

static void reply(int fd, int code, const char* ctype, const std::string& body) {
    char h[384];
    snprintf(h, sizeof(h), "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nCache-Control: no-cache, no-store, must-revalidate\r\nPragma: no-cache\r\nExpires: 0\r\nConnection: close\r\n\r\n",
             code, code == 200 ? "OK" : "Error", ctype, body.size());
    send_str(fd, h);
    send_str(fd, body);
}

bool WebServer::start() {
    lfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd_ < 0) return false;
    int one = 1;
    setsockopt(lfd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(s_.cfg->web_port);
    if (inet_pton(AF_INET, s_.cfg->web_bind.c_str(), &a.sin_addr) != 1) a.sin_addr.s_addr = INADDR_ANY;
    if (bind(lfd_, (sockaddr*)&a, sizeof(a)) < 0 || listen(lfd_, 16) < 0) {
        fprintf(stderr, "[web] không bind được %s:%d\n", s_.cfg->web_bind.c_str(), s_.cfg->web_port);
        close(lfd_); lfd_ = -1;
        return false;
    }
    run_ = true;
    th_ = std::thread([this] { accept_loop(); });
    fprintf(stderr, "[web] http://%s:%d\n", s_.cfg->web_bind.c_str(), s_.cfg->web_port);
    return true;
}

void WebServer::stop() {
    run_ = false;
    if (lfd_ >= 0) { shutdown(lfd_, SHUT_RDWR); close(lfd_); lfd_ = -1; }
    if (th_.joinable()) th_.join();
    for (int i = 0; i < 50 && clients_ > 0; i++) std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

void WebServer::accept_loop() {
    while (run_) {
        int fd = accept(lfd_, nullptr, nullptr);
        if (fd < 0) { if (!run_) break; continue; }
        if (clients_ >= 16) { close(fd); continue; }
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        timeval tv{0, 200000}; // 200ms timeout
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        clients_++;
        std::thread([this, fd] { handle(fd); close(fd); clients_--; }).detach();
    }
}

static float qparam(const std::string& q, const char* key, float def) {
    auto p = q.find(std::string(key) + "=");
    if (p == std::string::npos) return def;
    return (float)atof(q.c_str() + p + strlen(key) + 1);
}

void WebServer::handle(int fd) {
    char buf[32768];
    ssize_t n = recv(fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) return;
    buf[n] = 0;
    char method[8] = {0}, target[512] = {0};
    if (sscanf(buf, "%7s %511s", method, target) != 2) return;
    std::string path = target, query;
    auto qp = path.find('?');
    if (qp != std::string::npos) { query = path.substr(qp + 1); path = path.substr(0, qp); }

    if (path == "/" || path == "/index.html") {
        std::ifstream f(s_.cfg->www_dir + "/index.html", std::ios::binary);
        if (!f) { reply(fd, 404, "text/plain", "www/index.html not found"); return; }
        std::stringstream ss; ss << f.rdbuf();
        reply(fd, 200, "text/html; charset=utf-8", ss.str());
    } else if (path == "/video") {
        serve_mjpeg(fd);
    } else if (path == "/events") {
        serve_events(fd);
    } else if (path == "/api/select_box" && std::string(method) == "POST") {
        float x1 = qparam(query, "x1", -1), y1 = qparam(query, "y1", -1);
        float x2 = qparam(query, "x2", -1), y2 = qparam(query, "y2", -1);
        if (x1 < 0 || x1 > 1 || y1 < 0 || y1 > 1 || x2 < 0 || x2 > 1 || y2 < 0 || y2 > 1) {
            reply(fd, 400, "application/json", "{\"ok\":false,\"err\":\"invalid_coords\"}");
            return;
        }
        s_.ctl->reset();
        s_.mav->sync_target_to_current();
        s_.trk->select_box(x1, y1, x2, y2);
        reply(fd, 200, "application/json", "{\"ok\":true}");
    } else if (path == "/api/cancel" && std::string(method) == "POST") {
        s_.trk->cancel();
        s_.ctl->reset();
        s_.mav->send_rate(0, 0);
        s_.mav->sync_target_to_current();
        reply(fd, 200, "application/json", "{\"ok\":true}");
    } else if ((path == "/api/space" || path == "/api/recenter") && std::string(method) == "POST") {
        s_.trk->cancel();
        s_.ctl->reset();
        s_.mav->recenter();
        {
            std::ofstream f("/dev/shm/gesture_off");
            f << "OFF";
        }
        {
            std::lock_guard<std::mutex> l(gesture_m_);
            if (!latest_gesture_json_.empty()) {
                try {
                    auto j = json::parse(latest_gesture_json_);
                    j["mode"] = "OFF";
                    latest_gesture_json_ = j.dump();
                } catch (...) {}
            }
        }
        reply(fd, 200, "application/json", "{\"ok\":true}");
    } else if (path == "/api/status") {
        reply(fd, 200, "application/json", overlay_json());
    } else if (path == "/frame.jpg" || path == "/api/snapshot") {
        auto j = s_.jpeg->get();
        if (!j || j->data.empty()) {
            reply(fd, 503, "text/plain", "no frame");
            return;
        }
        std::string h = "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                        std::to_string(j->data.size()) + "\r\nConnection: close\r\n\r\n";
        send_str(fd, h);
        send_all(fd, j->data.data(), j->data.size());
        return;
    } else if (path.rfind("/api/face_crop", 0) == 0) {
        int id = (int)qparam(query, "id", 0);
        std::string crop_path = "/dev/shm/face_crop_" + std::to_string(id) + ".jpg";
        std::ifstream f(crop_path, std::ios::binary);
        if (!f) {
            f.open("/dev/shm/face_crop.jpg", std::ios::binary);
        }
        if (!f) {
            reply(fd, 404, "text/plain", "no face crop");
            return;
        }
        std::stringstream ss; ss << f.rdbuf();
        std::string d = ss.str();
        std::string h = "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\nContent-Length: " +
                        std::to_string(d.size()) + "\r\nConnection: close\r\n\r\n";
        send_str(fd, h);
        send_all(fd, d.data(), d.size());
        return;
    } else if (path == "/api/face_result" && std::string(method) == "POST") {
        const char* body = strstr(buf, "\r\n\r\n");
        if (body) {
            body += 4;
            std::lock_guard<std::mutex> l(face_m_);
            latest_face_json_ = body;
            latest_face_t_ = now_s();
        }
        reply(fd, 200, "application/json", "{\"ok\":true}");
    } else if (path == "/api/manual" && std::string(method) == "POST") {
        float yaw = qparam(query, "yaw", 0.0f);
        float pitch = qparam(query, "pitch", 0.0f);
        float ttl = qparam(query, "ttl", 150.0f);
        s_.ctl->set_manual(yaw, pitch, (int)ttl);
        reply(fd, 200, "application/json", "{\"ok\":true}");
    } else if (path == "/api/track_hand" && std::string(method) == "POST") {
        float x = qparam(query, "x", -1.0f);
        float y = qparam(query, "y", -1.0f);
        float ttl = qparam(query, "ttl", 180.0f);
        if (x >= 0 && y >= 0) {
            s_.ctl->set_hand_target(x, y, (int)ttl);
        }
        reply(fd, 200, "application/json", "{\"ok\":true}");
    } else if (path == "/api/gesture_result" && std::string(method) == "POST") {
        const char* body = strstr(buf, "\r\n\r\n");
        if (body) {
            body += 4;
            std::lock_guard<std::mutex> l(gesture_m_);
            latest_gesture_json_ = body;
            latest_gesture_t_ = now_s();
        }
        reply(fd, 200, "application/json", "{\"ok\":true}");
    } else {
        reply(fd, 404, "text/plain", "not found");
    }
}

void WebServer::serve_mjpeg(int fd) {
    const char* h = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    if (!send_str(fd, h)) return;
    uint64_t last = 0;
    while (run_) {
        auto j = s_.jpeg->get();
        if (!j || j->seq == last) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }

        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = POLLOUT;
        int ret = poll(&pfd, 1, 0);
        if (ret < 0) return;
        if (ret == 0 || !(pfd.revents & POLLOUT)) {
            // Client chưa đọc kịp frame trước, skip frame này để chống dồn đệm và giật đứng hình
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        last = j->seq;
        char ph[128];
        snprintf(ph, sizeof(ph), "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n", j->data.size());
        if (!send_str(fd, ph) || !send_all(fd, j->data.data(), j->data.size()) || !send_str(fd, "\r\n")) return;
    }
}

std::string WebServer::overlay_json() {
    json o;
    auto d = s_.det_res->get();
    auto t = s_.trk_res->get();
    double now = now_s();
    o["w"] = s_.cap->width(); o["h"] = s_.cap->height();
    json da = json::array();
    if (d) {
        for (auto& x : d->dets)
            da.push_back({{"b", {x.box.x1, x.box.y1, x.box.x2, x.box.y2}}, {"c", x.cls}, {"n", x.name}, {"s", x.score}});
        o["det_frame"] = d->frame_id;
        o["lat_ms"] = (now - d->t_capture) * 1000;
        o["dw"] = d->w; o["dh"] = d->h;
    }
    o["det"] = da;

    json tj;
    bool is_hand = s_.ctl->hand_tracking_active.load();
    tj["state"] = is_hand ? "HAND_TRACKING" : (t ? to_str(t->state) : "IDLE");
    tj["is_hand"] = is_hand;
    if (t && t->valid() && !is_hand) {
        tj["b"] = {t->box.x1, t->box.y1, t->box.x2, t->box.y2};
        tj["c"] = {t->cx, t->cy};
        tj["s"] = t->score;
        tj["tw"] = t->w; tj["th"] = t->h;
    }
    o["trk"] = tj;

    o["fps"] = {{"cap", s_.cap->fps.fps}, {"det", s_.det ? s_.det->fps.fps : 0.0}, {"trk", s_.trk->fps.fps}};
    o["infer_ms"] = s_.det ? s_.det->avg_infer_ms.load() : 0.0;
    o["rtsp"] = s_.cap->connected();
    o["mav"] = s_.mav->connected();
    o["gimbal"] = {{"yaw", s_.mav->yaw_deg()}, {"pitch", s_.mav->pitch_deg()},
                   {"ryaw", s_.ctl->out_yaw.load()}, {"rpitch", s_.ctl->out_pitch.load()}};

    {
        // 1. Đọc kết quả nhận diện khuôn mặt từ RAM SHM siêu tốc
        bool face_loaded = false;
        std::ifstream ff("/dev/shm/face_result.json");
        if (ff) {
            try {
                json fj;
                ff >> fj;
                double ts = fj.value("ts", 0.0);
                uint64_t now_ms = (uint64_t)(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
                if (now_ms >= ts && (now_ms - ts) <= 1200) {
                    o["face"] = fj;
                    face_loaded = true;
                }
            } catch (...) {}
        }
        if (!face_loaded) {
            std::lock_guard<std::mutex> l(face_m_);
            if (now - latest_face_t_ <= 1.2 && !latest_face_json_.empty()) {
                try {
                    o["face"] = json::parse(latest_face_json_);
                    face_loaded = true;
                } catch (...) {}
            }
        }
        if (!face_loaded) o["face"] = nullptr;
    }
    {
        // 2. Đọc kết quả nhận diện cử chỉ từ RAM SHM siêu tốc
        bool gesture_loaded = false;
        std::ifstream gf("/dev/shm/gesture_result.json");
        if (gf) {
            try {
                json gj;
                gf >> gj;
                double ts = gj.value("ts", 0.0);
                double now_sys = (double)time(nullptr);
                if (std::abs(now_sys - ts) <= 2.0) {
                    o["gesture"] = gj;
                    gesture_loaded = true;
                }
            } catch (...) {}
        }
        if (!gesture_loaded) {
            std::lock_guard<std::mutex> l(gesture_m_);
            if (now - latest_gesture_t_ <= 2.0 && !latest_gesture_json_.empty()) {
                try {
                    o["gesture"] = json::parse(latest_gesture_json_);
                    gesture_loaded = true;
                } catch (...) {}
            }
        }
        if (!gesture_loaded) o["gesture"] = nullptr;
    }
    return o.dump();
}

void WebServer::serve_events(int fd) {
    const char* h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\n\r\n";
    if (!send_str(fd, h)) return;
    double period = 1.0 / std::max(1, s_.cfg->overlay_hz);
    while (run_) {
        std::string msg = "data: " + overlay_json() + "\n\n";
        if (!send_str(fd, msg)) return;
        std::this_thread::sleep_for(std::chrono::duration<double>(period));
    }
}

}  // namespace gv
