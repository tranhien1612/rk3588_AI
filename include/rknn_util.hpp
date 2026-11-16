// Tiện ích RKNN + RGA dùng chung
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"
#include "im2d.h"
#include "rga.h"
#include "rknn_api.h"

namespace gv {

inline std::vector<uint8_t> read_file(const std::string& p) {
    std::vector<uint8_t> d;
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return d;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    d.resize(n);
    if (fread(d.data(), 1, n, f) != (size_t)n) d.clear();
    fclose(f);
    return d;
}

inline rknn_core_mask core_mask_of(int c) {
    switch (c) {
        case 0: return RKNN_NPU_CORE_0;
        case 1: return RKNN_NPU_CORE_1;
        case 2: return RKNN_NPU_CORE_2;
        default: return RKNN_NPU_CORE_AUTO;
    }
}

// NV12 -> RGB888 (RGA, 1 bước cả đổi màu lẫn resize) vào vùng nw*nh trong buffer rộng wstride.
// dst trỏ tới pixel đầu của vùng.
inline bool rga_nv12_to_rgb(const Frame& f, uint8_t* dst, int nw, int nh, int wstride) {
    rga_buffer_t src = wrapbuffer_virtualaddr_t((void*)f.nv12.data(), f.w, f.h, f.stride, f.h, RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t d = wrapbuffer_virtualaddr_t((void*)dst, nw, nh, wstride, nh, RK_FORMAT_RGB_888);
    rga_buffer_t pat;
    memset(&pat, 0, sizeof(pat));
    im_rect r0, r1, r2;
    memset(&r0, 0, sizeof(r0)); memset(&r1, 0, sizeof(r1)); memset(&r2, 0, sizeof(r2));
    IM_STATUS st = improcess(src, d, pat, r0, r1, r2, IM_SYNC);
    if (st != IM_STATUS_SUCCESS) {
        static std::atomic<int> warned{0};
        if (warned++ < 5) fprintf(stderr, "[rga] improcess lỗi: %s\n", imStrError(st));
        return false;
    }
    return true;
}

// Một ngữ cảnh RKNN + thông tin tensor
struct RknnCtx {
    rknn_context ctx = 0;
    rknn_input_output_num io{};
    std::vector<rknn_tensor_attr> in_attr, out_attr;
    int in_w = 0, in_h = 0, in_c = 3;
    std::vector<int32_t> zps;
    std::vector<float> scales;

    // parent != nullptr -> rknn_dup_context (chia sẻ weight)
    bool init(const std::vector<uint8_t>& model, const rknn_context* parent, int core) {
        int ret;
        if (parent) ret = rknn_dup_context(const_cast<rknn_context*>(parent), &ctx);
        else ret = rknn_init(&ctx, (void*)model.data(), model.size(), 0, nullptr);
        if (ret < 0) { fprintf(stderr, "[rknn] init lỗi %d\n", ret); return false; }
        if (core >= 0) {
            ret = rknn_set_core_mask(ctx, core_mask_of(core));
            if (ret < 0) fprintf(stderr, "[rknn] set_core_mask(%d) lỗi %d\n", core, ret);
        }
        if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) < 0) return false;
        in_attr.assign(io.n_input, rknn_tensor_attr{});
        out_attr.assign(io.n_output, rknn_tensor_attr{});
        for (uint32_t i = 0; i < io.n_input; i++) {
            in_attr[i].index = i;
            rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr[i], sizeof(rknn_tensor_attr));
        }
        for (uint32_t i = 0; i < io.n_output; i++) {
            out_attr[i].index = i;
            rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr[i], sizeof(rknn_tensor_attr));
            zps.push_back(out_attr[i].zp);
            scales.push_back(out_attr[i].scale);
        }
        if (in_attr[0].fmt == RKNN_TENSOR_NCHW) { in_c = in_attr[0].dims[1]; in_h = in_attr[0].dims[2]; in_w = in_attr[0].dims[3]; }
        else { in_h = in_attr[0].dims[1]; in_w = in_attr[0].dims[2]; in_c = in_attr[0].dims[3]; }
        return true;
    }
    void destroy() { if (ctx) { rknn_destroy(ctx); ctx = 0; } }
};

}  // namespace gv

