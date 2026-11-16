#include "config.hpp"

#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace gv {
using nlohmann::json;

template <class T>
static void get(const json& j, const char* sec, const char* key, T& out) {
    if (!j.contains(sec)) return;
    const json& s = j[sec];
    if (!s.contains(key)) return;
    try {
        out = s[key].get<T>();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("config: sai kiểu '") + sec + "." + key + "': " + e.what());
    }
}

Config Config::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("config: không mở được file " + path);
    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("config: JSON lỗi: ") + e.what());
    }
    Config c;
    get(j, "camera", "rtsp_url", c.rtsp_url);
    get(j, "camera", "transport", c.transport);
    get(j, "camera", "codec", c.codec);
    get(j, "camera", "latency_ms", c.latency_ms);
    get(j, "camera", "file_realtime", c.file_realtime);
    get(j, "camera", "jpeg_quant", c.jpeg_quant);
    get(j, "camera", "reconnect_ms", c.reconnect_ms);

    get(j, "mavlink", "enabled", c.mav_enabled);
    get(j, "mavlink", "endpoint", c.mav_port);
    get(j, "mavlink", "baud", c.mav_baud);
    get(j, "mavlink", "sys_id", c.mav_sys);
    get(j, "mavlink", "comp_id", c.mav_comp);
    get(j, "mavlink", "target_sys", c.mav_target_sys);
    get(j, "mavlink", "target_comp", c.mav_target_comp);

    get(j, "control", "loop_hz", c.ctl_hz);
    get(j, "control", "deadband", c.deadband);
    get(j, "control", "max_rate", c.max_rate);
    get(j, "control", "max_accel", c.max_accel);
    get(j, "control", "ff_gain", c.ff_gain);
    get(j, "control", "kp_yaw", c.kp_yaw);
    get(j, "control", "ki_yaw", c.ki_yaw);
    get(j, "control", "kd_yaw", c.kd_yaw);
    get(j, "control", "invert_yaw", c.invert_yaw);
    get(j, "control", "kp_pitch", c.kp_pitch);
    get(j, "control", "ki_pitch", c.ki_pitch);
    get(j, "control", "kd_pitch", c.kd_pitch);
    get(j, "control", "invert_pitch", c.invert_pitch);
    get(j, "control", "lost_timeout_ms", c.lost_timeout_ms);
    get(j, "control", "max_yaw_deg", c.max_yaw_deg);
    get(j, "control", "min_pitch_deg", c.min_pitch_deg);
    get(j, "control", "max_pitch_deg", c.max_pitch_deg);

    get(j, "detector", "enabled", c.det_enabled);
    get(j, "detector", "model", c.det_model);
    get(j, "detector", "labels", c.det_labels);
    get(j, "detector", "conf_thresh", c.det_conf);
    get(j, "detector", "nms_thresh", c.det_nms);
    get(j, "detector", "workers", c.det_workers);
    get(j, "detector", "npu_cores", c.det_cores);

    get(j, "tracker", "backbone_z", c.trk_backbone_z);
    get(j, "tracker", "backbone", c.trk_backbone_x);
    get(j, "tracker", "head", c.trk_head);
    get(j, "tracker", "min_score", c.trk_min_score);

    get(j, "web", "bind", c.web_bind);
    get(j, "web", "port", c.web_port);
    get(j, "web", "overlay_hz", c.overlay_hz);
    get(j, "web", "www_dir", c.www_dir);

    get(j, "system", "cpu_affinity_ai", c.aff_ai);
    get(j, "system", "cpu_affinity_io", c.aff_io);

    // kiểm tra hợp lệ
    if (c.rtsp_url.empty()) throw std::runtime_error("config: camera.rtsp_url rỗng");
    if (c.transport != "tcp" && c.transport != "udp") throw std::runtime_error("config: camera.transport phải là tcp|udp");
    if (c.codec != "h264" && c.codec != "h265") throw std::runtime_error("config: camera.codec phải là h264|h265");
    if (c.det_workers < 1 || c.det_workers > 12) throw std::runtime_error("config: detector.workers 1..12");
    if (c.det_cores.empty()) throw std::runtime_error("config: detector.npu_cores rỗng");
    if (c.ctl_hz < 5 || c.ctl_hz > 200) throw std::runtime_error("config: control.loop_hz 5..200");
    if (c.web_port < 1 || c.web_port > 65535) throw std::runtime_error("config: web.port sai");
    return c;
}

}  // namespace gv
