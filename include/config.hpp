// Cấu hình đọc 1 lần từ config/config.json (cố định khi chạy)
#pragma once
#include <string>
#include <vector>

namespace gv {

struct Config {
    // camera
    std::string rtsp_url = "rtsp://192.168.144.108";
    std::string transport = "tcp";   // tcp | udp
    std::string codec = "h264";      // h264 | h265
    int latency_ms = 0;
    bool file_realtime = true;       // chỉ khi nguồn là file (test): phát đúng tốc độ
    int jpeg_quant = 7;              // mppjpegenc quant 0..10 (cao = chất lượng tốt)
    int reconnect_ms = 2000;
    // mavlink
    std::string mav_port = "/dev/ttyUSB0";
    int mav_baud = 115200;
    int mav_sys = 255, mav_comp = 190, mav_target_sys = 1, mav_target_comp = 154;
    bool mav_enabled = true;
    // control
    int ctl_hz = 30;
    double deadband = 0.04, max_rate = 0.8, max_accel = 3.0, ff_gain = 0.3;
    double kp_yaw = 0.6, ki_yaw = 0.0, kd_yaw = 0.1; bool invert_yaw = true;
    double kp_pitch = 0.6, ki_pitch = 0.0, kd_pitch = 0.1; bool invert_pitch = true;
    int lost_timeout_ms = 500;
    double max_yaw_deg = 170, min_pitch_deg = -90, max_pitch_deg = 30;
    // detector
    std::string det_model = "models/yolov5s-640-640.rknn";
    std::string det_labels = "models/coco_80_labels_list.txt";
    float det_conf = 0.5f, det_nms = 0.45f;
    int det_workers = 3;
    std::vector<int> det_cores = {0, 1, 2};
    bool det_enabled = true;
    // tracker (NanoTrack)
    std::string trk_backbone_z = "models/nanotrack_backbone_127.onnx";
    std::string trk_backbone_x = "models/nanotrack_backbone.onnx";
    std::string trk_head = "models/nanotrack_head.onnx";
    float trk_min_score = 0.25f;
    // web
    std::string web_bind = "0.0.0.0";
    int web_port = 8080;
    int overlay_hz = 30;
    std::string www_dir = "www";
    // system
    std::vector<int> aff_ai = {4, 5, 6, 7};
    std::vector<int> aff_io = {0, 1, 2, 3};

    // Nạp file; ném std::runtime_error với thông báo rõ ràng nếu lỗi.
    static Config load(const std::string& path);
};

}  // namespace gv
