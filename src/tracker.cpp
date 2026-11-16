#include "tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace gv {

// ---------------- Implementation của NanoTrack chuẩn xác theo OpenCV TrackerNano (như z1Camera_mavlink) ----------------
class NanoTrackBackend {
public:
    NanoTrackBackend(const std::string& backbone_z_path,
                     const std::string& backbone_x_path,
                     const std::string& head_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "nanotrack") {
        session_options_.SetIntraOpNumThreads(2);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_z_ = std::make_unique<Ort::Session>(env_, backbone_z_path.c_str(), session_options_);
        session_x_ = std::make_unique<Ort::Session>(env_, backbone_x_path.c_str(), session_options_);
        session_head_ = std::make_unique<Ort::Session>(env_, head_path.c_str(), session_options_);

        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        crop_z_chw_.resize(3 * exemplar_size_ * exemplar_size_);
        crop_x_chw_.resize(3 * instance_size_ * instance_size_);
        z_features_.resize(48 * 8 * 8);

        initGrids();
    }

    void init(const Frame& frame, const Box& bbox) {
        float bw = std::max(10.0f, bbox.x2 - bbox.x1);
        float bh = std::max(10.0f, bbox.y2 - bbox.y1);
        target_pos_[0] = (bbox.x1 + bbox.x2) * 0.5f;
        target_pos_[1] = (bbox.y1 + bbox.y2) * 0.5f;
        target_sz_[0] = bw;
        target_sz_[1] = bh;

        float sum_sz = target_sz_[0] + target_sz_[1];
        float w_extent = target_sz_[0] + context_amount_ * sum_sz;
        float h_extent = target_sz_[1] + context_amount_ * sum_sz;
        int sz = (int)std::sqrt(w_extent * h_extent);

        getSubwindow(frame, target_pos_[0], target_pos_[1], sz, exemplar_size_, crop_z_chw_);

        // Run backbone Z (1x3x127x127) -> z_features (1x48x8x8)
        std::vector<int64_t> in_shape = {1, 3, exemplar_size_, exemplar_size_};
        auto in_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, crop_z_chw_.data(), crop_z_chw_.size(), in_shape.data(), in_shape.size());

        const char* in_names[] = {"input"};
        const char* out_names[] = {"output"};
        auto out_tensors = session_z_->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);

        const float* z_data = out_tensors[0].GetTensorData<float>();
        std::memcpy(z_features_.data(), z_data, 48 * 8 * 8 * sizeof(float));
    }

    bool update(const Frame& frame, Box& out_box, float& score) {
        int target_sz_sum = (int)(target_sz_[0] + target_sz_[1]);
        float wc = target_sz_[0] + context_amount_ * target_sz_sum;
        float hc = target_sz_[1] + context_amount_ * target_sz_sum;
        float sz = std::sqrt(wc * hc);
        float scale_z = (float)exemplar_size_ / sz;
        float sx = sz * ((float)instance_size_ / (float)exemplar_size_);

        float target_sz_scaled_w = target_sz_[0] * scale_z;
        float target_sz_scaled_h = target_sz_[1] * scale_z;

        getSubwindow(frame, target_pos_[0], target_pos_[1], (int)sx, instance_size_, crop_x_chw_);

        // Run backbone X (1x3x255x255) -> x_features (1x48x16x16)
        std::vector<int64_t> in_x_shape = {1, 3, instance_size_, instance_size_};
        auto in_x_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, crop_x_chw_.data(), crop_x_chw_.size(), in_x_shape.data(), in_x_shape.size());

        const char* in_x_names[] = {"input"};
        const char* out_x_names[] = {"output"};
        auto out_x_tensors = session_x_->Run(Ort::RunOptions{nullptr}, in_x_names, &in_x_tensor, 1, out_x_names, 1);

        // Run Head: input1 (z: 1x48x8x8), input2 (x: 1x48x16x16)
        std::vector<int64_t> z_shape = {1, 48, 8, 8};
        auto z_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, z_features_.data(), z_features_.size(), z_shape.data(), z_shape.size());

        const char* in_head_names[] = {"input1", "input2"};
        const char* out_head_names[] = {"output1", "output2"};
        Ort::Value head_inputs[2] = {std::move(z_tensor), std::move(out_x_tensors[0])};

        auto head_outputs = session_head_->Run(Ort::RunOptions{nullptr}, in_head_names, head_inputs, 2, out_head_names, 2);

        // Output1: Cls Score (1x2x16x16), Output2: BBox Pred (1x4x16x16)
        const float* cls_data = head_outputs[0].GetTensorData<float>();
        const float* bbox_data = head_outputs[1].GetTensorData<float>();

        int total_pixels = score_size_ * score_size_;
        int best_idx = 0;
        float max_pscore = -1e9f;
        float best_pred_x1 = 0, best_pred_y1 = 0, best_pred_x2 = 0, best_pred_y2 = 0;
        float best_penalty = 1.0f;
        float best_score = 0.0f;

        float ratio_val = target_sz_scaled_w / target_sz_scaled_h;
        float target_sz_cal = sizeCal(target_sz_scaled_w, target_sz_scaled_h);

        for (int i = 0; i < total_pixels; i++) {
            float c0 = cls_data[i];
            float c1 = cls_data[total_pixels + i];
            float max_c = std::max(c0, c1);
            float exp0 = std::exp(c0 - max_c);
            float exp1 = std::exp(c1 - max_c);
            float s1 = exp1 / (exp0 + exp1);

            float dx1 = bbox_data[0 * total_pixels + i];
            float dy1 = bbox_data[1 * total_pixels + i];
            float dx2 = bbox_data[2 * total_pixels + i];
            float dy2 = bbox_data[3 * total_pixels + i];

            float px1 = grid_x_[i] - dx1;
            float py1 = grid_y_[i] - dy1;
            float px2 = grid_x_[i] + dx2;
            float py2 = grid_y_[i] + dy2;

            float pw = px2 - px1;
            float ph = py2 - py1;

            float sc = sizeCal(pw, ph) / target_sz_cal;
            sc = std::max(sc, 1.0f / sc);

            float rc = ratio_val / (pw / ph);
            rc = std::max(rc, 1.0f / rc);

            float penalty = std::exp(-(rc * sc - 1.0f) * penalty_k_);
            float pscore = penalty * s1;

            pscore = pscore * (1.0f - window_influence_) + hanning_[i] * window_influence_;

            if (pscore > max_pscore) {
                max_pscore = pscore;
                best_idx = i;
                best_penalty = penalty;
                best_score = s1;
                best_pred_x1 = px1;
                best_pred_y1 = py1;
                best_pred_x2 = px2;
                best_pred_y2 = py2;
            }
        }

        score = max_pscore;

        float pred_xs = (best_pred_x1 + best_pred_x2) * 0.5f;
        float pred_ys = (best_pred_y1 + best_pred_y2) * 0.5f;
        float pred_w = (best_pred_x2 - best_pred_x1) / scale_z;
        float pred_h = (best_pred_y2 - best_pred_y1) / scale_z;

        float diff_xs = (pred_xs - instance_size_ / 2.0f) / scale_z;
        float diff_ys = (pred_ys - instance_size_ / 2.0f) / scale_z;

        float lr = best_penalty * best_score * lr_;

        float res_x = target_pos_[0] + diff_xs;
        float res_y = target_pos_[1] + diff_ys;
        float res_w = pred_w * lr + (1.0f - lr) * target_sz_[0];
        float res_h = pred_h * lr + (1.0f - lr) * target_sz_[1];

        res_x = std::max(0.0f, std::min((float)frame.w, res_x));
        res_y = std::max(0.0f, std::min((float)frame.h, res_y));
        res_w = std::max(10.0f, std::min((float)frame.w, res_w));
        res_h = std::max(10.0f, std::min((float)frame.h, res_h));

        target_pos_[0] = res_x;
        target_pos_[1] = res_y;
        target_sz_[0] = res_w;
        target_sz_[1] = res_h;

        out_box.x1 = res_x - res_w * 0.5f;
        out_box.y1 = res_y - res_h * 0.5f;
        out_box.x2 = res_x + res_w * 0.5f;
        out_box.y2 = res_y + res_h * 0.5f;

        return true;
    }

private:
    float sizeCal(float w, float h) {
        float pad = (w + h) * 0.5f;
        return std::sqrt((w + pad) * (h + pad));
    }

    void initGrids() {
        score_size_ = 16;
        grid_x_.resize(score_size_ * score_size_);
        grid_y_.resize(score_size_ * score_size_);
        hanning_.resize(score_size_ * score_size_);

        int sz2 = score_size_ / 2;
        std::vector<float> hanning_1d(score_size_);
        for (int i = 0; i < score_size_; i++) {
            hanning_1d[i] = 0.5f - 0.5f * std::cos(2.0f * M_PI * (i + 0.5f) / (float)score_size_);
        }

        for (int r = 0; r < score_size_; r++) {
            for (int c = 0; c < score_size_; c++) {
                int idx = r * score_size_ + c;
                grid_x_[idx] = (c - sz2) * total_stride_ + instance_size_ / 2.0f;
                grid_y_[idx] = (r - sz2) * total_stride_ + instance_size_ / 2.0f;
                hanning_[idx] = hanning_1d[r] * hanning_1d[c];
            }
        }
    }

    // getSubwindow chuẩn của OpenCV TrackerNano
    void getSubwindow(const Frame& frame, float pos_x, float pos_y, int original_sz, int resize_sz, std::vector<float>& dst_crop) {
        const uint8_t* Y = frame.y();
        const uint8_t* UV = frame.nv12.data() + (size_t)frame.stride * frame.h;

        int c = (original_sz + 1) / 2;
        float scale = (float)original_sz / (float)resize_sz;
        int plane = resize_sz * resize_sz;
        float* out_r = dst_crop.data();
        float* out_g = dst_crop.data() + plane;
        float* out_b = dst_crop.data() + 2 * plane;

        for (int r = 0; r < resize_sz; r++) {
            float sy = (pos_y - c) + (r + 0.5f) * scale;
            for (int col = 0; col < resize_sz; col++) {
                float sx = (pos_x - c) + (col + 0.5f) * scale;
                int idx = r * resize_sz + col;

                if (sx < 0 || sx >= frame.w - 1 || sy < 0 || sy >= frame.h - 1) {
                    out_r[idx] = 128.0f;
                    out_g[idx] = 128.0f;
                    out_b[idx] = 128.0f;
                    continue;
                }

                int x0 = (int)sx;
                int y0 = (int)sy;
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                float wx1 = sx - (float)x0;
                float wx0 = 1.0f - wx1;
                float wy1 = sy - (float)y0;
                float wy0 = 1.0f - wy1;

                float y_val = wy0 * (wx0 * Y[y0 * frame.stride + x0] + wx1 * Y[y0 * frame.stride + x1]) +
                              wy1 * (wx0 * Y[y1 * frame.stride + x0] + wx1 * Y[y1 * frame.stride + x1]);

                int uv_y = (int)(sy * 0.5f);
                int uv_x = ((int)(sx * 0.5f)) * 2;
                uv_y = std::clamp(uv_y, 0, frame.h / 2 - 1);
                uv_x = std::clamp(uv_x, 0, frame.stride - 2);

                float u_val = (float)UV[uv_y * frame.stride + uv_x] - 128.0f;
                float v_val = (float)UV[uv_y * frame.stride + uv_x + 1] - 128.0f;

                float r_val = y_val + 1.402f * v_val;
                float g_val = y_val - 0.344136f * u_val - 0.714136f * v_val;
                float b_val = y_val + 1.772f * u_val;

                out_r[idx] = std::clamp(r_val, 0.0f, 255.0f);
                out_g[idx] = std::clamp(g_val, 0.0f, 255.0f);
                out_b[idx] = std::clamp(b_val, 0.0f, 255.0f);
            }
        }
    }

    const int exemplar_size_ = 127;
    const int instance_size_ = 255;
    const int total_stride_ = 16;
    int score_size_ = 16;

    const float context_amount_ = 0.5f;
    const float window_influence_ = 0.455f;
    const float lr_ = 0.37f;
    const float penalty_k_ = 0.055f;

    float target_pos_[2] = {0, 0};
    float target_sz_[2] = {0, 0};

    std::vector<float> grid_x_;
    std::vector<float> grid_y_;
    std::vector<float> hanning_;
    std::vector<float> z_features_;
    std::vector<float> crop_z_chw_;
    std::vector<float> crop_x_chw_;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_z_;
    std::unique_ptr<Ort::Session> session_x_;
    std::unique_ptr<Ort::Session> session_head_;
    Ort::MemoryInfo memory_info_{nullptr};
};

// ---------------- TrackerManager ----------------
TrackerManager::TrackerManager(const Config& cfg, FrameMailbox& mb, Latest<TrackResult>& out)
    : cfg_(cfg), mb_(mb), out_(out) {
    try {
        nano_ = std::make_unique<NanoTrackBackend>(cfg.trk_backbone_z, cfg.trk_backbone_x, cfg.trk_head);
        fprintf(stderr, "[tracker] NanoTrack ONNX backend loaded successfully.\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "[tracker] Lỗi nạp NanoTrack ONNX: %s\n", e.what());
    }
}

TrackerManager::~TrackerManager() {
    stop();
}

void TrackerManager::start() {
    run_ = true;
    th_ = std::thread([this] { run(); });
}

void TrackerManager::stop() {
    run_ = false;
    if (th_.joinable()) th_.join();
}

void TrackerManager::select_box(float x1, float y1, float x2, float y2) {
    std::lock_guard<std::mutex> l(m_);
    has_sel_box_ = true;
    sel_x1_ = std::min(x1, x2);
    sel_y1_ = std::min(y1, y2);
    sel_x2_ = std::max(x1, x2);
    sel_y2_ = std::max(y1, y2);
}

void TrackerManager::cancel() {
    std::lock_guard<std::mutex> l(m_);
    do_cancel_ = true;
}

void TrackerManager::run() {
    TrackState st = TrackState::IDLE;

    uint64_t last_id = 0;
    double low_since = 0;

    // Bộ triệt tiêu rung giật (micro-jitter deadband)
    double stable_cx = 0, stable_cy = 0, stable_w = 0, stable_h = 0;
    double last_pub_cx = 0, last_pub_cy = 0;

    auto publish = [&](uint64_t id, double tc, int w, int h, float score) {
        auto r = std::make_shared<TrackResult>();
        r->state = st;
        r->frame_id = id;
        r->t_capture = tc;
        r->w = w;
        r->h = h;
        r->score = score;
        if (st == TrackState::TRACKING || st == TrackState::LOW_CONF) {
            r->cx = (float)stable_cx;
            r->cy = (float)stable_cy;
            r->vx = (float)((stable_cx - last_pub_cx) / 0.033);
            r->vy = (float)((stable_cy - last_pub_cy) / 0.033);
            r->box = {
                (float)(stable_cx - stable_w * 0.5),
                (float)(stable_cy - stable_h * 0.5),
                (float)(stable_cx + stable_w * 0.5),
                (float)(stable_cy + stable_h * 0.5)
            };
        }
        last_pub_cx = stable_cx;
        last_pub_cy = stable_cy;
        out_.set(r);
    };

    while (run_) {
        bool sel_box = false, cancel = false;
        float bx1 = 0, by1 = 0, bx2 = 0, by2 = 0;
        {
            std::lock_guard<std::mutex> l(m_);
            sel_box = has_sel_box_;
            cancel = do_cancel_;
            bx1 = sel_x1_; by1 = sel_y1_; bx2 = sel_x2_; by2 = sel_y2_;
            has_sel_box_ = false;
            do_cancel_ = false;
        }

        FramePtr f = mb_.wait_newer(last_id, 100);

        if (cancel) {
            st = TrackState::IDLE;
            auto fr = mb_.latest();
            publish(fr ? fr->id : last_id, fr ? fr->t_capture : now_s(), fr ? fr->w : 0, fr ? fr->h : 0, 0);
        }

        if (sel_box && nano_) {
            FramePtr cf = mb_.latest();
            if (cf) {
                Box b;
                b.x1 = bx1 * cf->w;
                b.y1 = by1 * cf->h;
                b.x2 = bx2 * cf->w;
                b.y2 = by2 * cf->h;

                if (b.x2 - b.x1 < 10.0f) { b.x1 -= 10.0f; b.x2 += 10.0f; }
                if (b.y2 - b.y1 < 10.0f) { b.y1 -= 10.0f; b.y2 += 10.0f; }
                b.x1 = std::max(0.0f, b.x1);
                b.y1 = std::max(0.0f, b.y1);
                b.x2 = std::min((float)cf->w, b.x2);
                b.y2 = std::min((float)cf->h, b.y2);

                nano_->init(*cf, b);

                double cx = (b.x1 + b.x2) * 0.5;
                double cy = (b.y1 + b.y2) * 0.5;
                double bw = b.x2 - b.x1;
                double bh = b.y2 - b.y1;

                stable_cx = last_pub_cx = cx;
                stable_cy = last_pub_cy = cy;
                stable_w = bw;
                stable_h = bh;

                low_since = 0;
                st = TrackState::TRACKING;
                last_id = cf->id;

                publish(cf->id, cf->t_capture, cf->w, cf->h, 1.0f);
                f = nullptr;
            }
        }

        if (!f || f->id <= last_id) continue;
        // Bắt kịp frame mới nhất nếu có frame mới hơn trong mailbox (loại bỏ độ trễ tích lũy)
        FramePtr lf = mb_.latest();
        if (lf && lf->id > f->id) {
            f = lf;
        }
        last_id = f->id;

        if (st != TrackState::TRACKING && st != TrackState::LOW_CONF) {
            publish(f->id, f->t_capture, f->w, f->h, 0);
            continue;
        }

        if (!nano_) {
            st = TrackState::LOST;
            publish(f->id, f->t_capture, f->w, f->h, 0);
            continue;
        }

        Box b;
        float score = 0;
        nano_->update(*f, b, score);

        if (score >= cfg_.trk_min_score) {
            double cx = (b.x1 + b.x2) * 0.5;
            double cy = (b.y1 + b.y2) * 0.5;
            double bw = b.x2 - b.x1;
            double bh = b.y2 - b.y1;

            // Kiểm tra sanity check giống y như z1Camera_mavlink
            // 1. Kích thước box hợp lệ
            if (bw < 8 || bh < 8 || bw > f->w * 0.85 || bh > f->h * 0.85) {
                st = TrackState::LOST;
                publish(f->id, f->t_capture, f->w, f->h, 0);
                continue;
            }

            // 2. Tâm phải nằm trong khung hình
            if (cx < 0 || cx >= f->w || cy < 0 || cy >= f->h) {
                st = TrackState::LOST;
                publish(f->id, f->t_capture, f->w, f->h, 0);
                continue;
            }

            // 3. Outlier check: ngăn chặn nhảy vị trí đột biến
            double jump_dist = std::hypot(cx - stable_cx, cy - stable_cy);
            double max_jump = std::hypot(f->w, f->h) * 0.28;
            if (jump_dist > max_jump && score < 0.65f) {
                st = TrackState::LOST;
                publish(f->id, f->t_capture, f->w, f->h, 0);
                continue;
            }

            // 4. Triệt tiêu dao động micro-jitter (khi vật thể đứng im, NanoTrack dao động vi mô < 3.0px)
            // Nếu độ dịch chuyển < 3.0px, cố định vị trí cũ để box không bị giật/nhảy lên xuống
            double dx = cx - stable_cx;
            double dy = cy - stable_cy;
            if (std::abs(dx) > 3.0) {
                stable_cx += (dx > 0 ? dx - 2.5 : dx + 2.5);
            }
            if (std::abs(dy) > 3.0) {
                stable_cy += (dy > 0 ? dy - 2.5 : dy + 2.5);
            }

            double dw = bw - stable_w;
            double dh = bh - stable_h;
            if (std::abs(dw) > 3.5) stable_w += (dw > 0 ? dw - 2.5 : dw + 2.5);
            if (std::abs(dh) > 3.5) stable_h += (dh > 0 ? dh - 2.5 : dh + 2.5);

            st = TrackState::TRACKING;
            low_since = 0;
        } else {
            st = TrackState::LOW_CONF;
            if (low_since == 0) low_since = f->t_capture;
            if ((f->t_capture - low_since) * 1000 > cfg_.lost_timeout_ms) {
                st = TrackState::LOST;
            }
        }

        publish(f->id, f->t_capture, f->w, f->h, score);
        fps.tick();
    }
}

}  // namespace gv
