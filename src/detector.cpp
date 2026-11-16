#include "detector.hpp"

#include <algorithm>

#include "rknn_util.hpp"
#include "yolo_postprocess.h"

namespace gv {

struct Detector::Impl {
    RknnCtx rk;
    std::vector<uint8_t> rgb;  // buffer input RGB 640x640 (viền pad = 114, giữ nguyên giữa các lần chạy)
    int cached_w = 0, cached_h = 0, nw = 0, nh = 0, padx = 0, pady = 0;
    float scale = 1;
};

bool Detector::start() {
    if (yolo_init_labels() != 0) { fprintf(stderr, "[det] không đọc được labels\n"); return false; }
    auto model = read_file(cfg_.det_model);
    if (model.empty()) { fprintf(stderr, "[det] không đọc được model %s\n", cfg_.det_model.c_str()); return false; }
    for (int i = 0; i < cfg_.det_workers; i++) {
        auto* im = new Impl();
        int core = cfg_.det_cores[i % cfg_.det_cores.size()];
        bool ok = im->rk.init(model, i == 0 ? nullptr : &impls_[0]->rk.ctx, core);
        if (!ok) { delete im; fprintf(stderr, "[det] init worker %d lỗi\n", i); return false; }
        if (im->rk.io.n_output != 3 || im->rk.in_c != 3) {
            fprintf(stderr, "[det] model không đúng dạng yolov5 3 đầu ra (n_output=%u)\n", im->rk.io.n_output);
            return false;
        }
        im->rgb.assign((size_t)im->rk.in_w * im->rk.in_h * 3, 114);
        impls_.push_back(im);
        if (i == 0) fprintf(stderr, "[det] model input %dx%d, workers=%d\n", im->rk.in_w, im->rk.in_h, cfg_.det_workers);
    }
    run_ = true;
    for (int i = 0; i < cfg_.det_workers; i++) ths_.emplace_back([this, i] { worker(i); });
    return true;
}

void Detector::stop() {
    run_ = false;
    for (auto& t : ths_) if (t.joinable()) t.join();
    ths_.clear();
    for (auto* im : impls_) { im->rk.destroy(); delete im; }
    impls_.clear();
}

void Detector::worker(int idx) {
    Impl& im = *impls_[idx];
    RknnCtx& rk = im.rk;
    const int mw = rk.in_w, mh = rk.in_h;
    rknn_input in[1];
    memset(in, 0, sizeof(in));
    in[0].index = 0;
    in[0].type = RKNN_TENSOR_UINT8;
    in[0].size = mw * mh * 3;
    in[0].fmt = RKNN_TENSOR_NHWC;
    in[0].pass_through = 0;
    in[0].buf = im.rgb.data();
    double ema = 0;
    while (run_) {
        FramePtr f = mb_.claim_latest(claimed_, 200);   // bỏ frame cũ: chỉ lấy frame mới nhất chưa ai xử lý
        if (!f) continue;
        double t0 = now_s();
        if (f->w != im.cached_w || f->h != im.cached_h) {
            // letterbox: tính lại vùng ảnh, tô lại viền
            im.scale = std::min((float)mw / f->w, (float)mh / f->h);
            im.nw = std::max(2, (int)(f->w * im.scale) & ~1);
            im.nh = std::max(2, (int)(f->h * im.scale) & ~1);
            im.padx = ((mw - im.nw) / 2) & ~1;
            im.pady = ((mh - im.nh) / 2) & ~1;
            std::fill(im.rgb.begin(), im.rgb.end(), 114);
            im.cached_w = f->w; im.cached_h = f->h;
        }
        uint8_t* dst = im.rgb.data() + ((size_t)im.pady * mw + im.padx) * 3;
        if (!rga_nv12_to_rgb(*f, dst, im.nw, im.nh, mw)) continue;

        if (idx == 0 && getenv("GV_DUMP_INPUT")) {   // debug: lưu ảnh đầu vào model (1 lần)
            static bool dumped = false;
            if (!dumped) { dumped = true; FILE* fp = fopen("/tmp/det_input.ppm", "wb");
                if (fp) { fprintf(fp, "P6\n%d %d\n255\n", mw, mh); fwrite(im.rgb.data(), 1, im.rgb.size(), fp); fclose(fp); }
                fprintf(stderr, "[det] dump input: frame %dx%d scale=%.3f nw=%d nh=%d padx=%d pady=%d\n", f->w, f->h, im.scale, im.nw, im.nh, im.padx, im.pady); }
        }
        rknn_inputs_set(rk.ctx, 1, in);
        if (rknn_run(rk.ctx, nullptr) < 0) continue;
        rknn_output out[3];
        memset(out, 0, sizeof(out));
        for (int i = 0; i < 3; i++) out[i].want_float = 0;
        if (rknn_outputs_get(rk.ctx, 3, out, nullptr) < 0) continue;

        detect_result_group_t grp;
        BOX_RECT pads{im.padx, 0, im.pady, 0};
        // post_process dùng pads.left / pads.top, scale_w/scale_h
        pads.left = im.padx; pads.top = im.pady;
        post_process((int8_t*)out[0].buf, (int8_t*)out[1].buf, (int8_t*)out[2].buf, mh, mw, cfg_.det_conf,
                     cfg_.det_nms, pads, im.scale, im.scale, rk.zps, rk.scales, &grp);
        rknn_outputs_release(rk.ctx, 3, out);

        auto r = std::make_shared<DetResult>();
        r->frame_id = f->id; r->t_capture = f->t_capture; r->w = f->w; r->h = f->h;
        for (int i = 0; i < grp.count; i++) {
            const auto& d = grp.results[i];
            Det x;
            x.box = {std::clamp((float)d.box.left, 0.f, (float)f->w), std::clamp((float)d.box.top, 0.f, (float)f->h),
                     std::clamp((float)d.box.right, 0.f, (float)f->w), std::clamp((float)d.box.bottom, 0.f, (float)f->h)};
            x.score = d.prop; x.cls = d.cls; x.name = d.name;
            r->dets.push_back(std::move(x));
        }
        r->t_done = now_s();
        {   // chỉ ghi nếu frame mới hơn kết quả đang có (worker có thể xong không theo thứ tự)
            std::lock_guard<std::mutex> l(pub_m_);
            if (r->frame_id > published_) { published_ = r->frame_id; out_.set(r); }
        }
        fps.tick();
        double ms = (r->t_done - t0) * 1000;
        ema = ema == 0 ? ms : ema * 0.95 + ms * 0.05;
        if (idx == 0) avg_infer_ms = ema;
    }
}

}  // namespace gv

