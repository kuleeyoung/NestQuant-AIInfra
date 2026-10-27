#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float round_even(float x) { return nearbyintf(x); }

__global__ void e8_nearest_kernel(const float* x, float* out, int64_t n) {
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const float* in = x + row * 8;
    float* dst = out + row * 8;
    float qi[8], qh[8], er[8], eh[8];
    float se_i = 0.0f, se_h = 0.0f;
    int parity_i = 0, parity_h = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        qi[j] = round_even(in[j]);
        qh[j] = round_even(in[j] - 0.5f) + 0.5f;
        er[j] = in[j] - qi[j];
        eh[j] = in[j] - qh[j];
        se_i += er[j] * er[j];
        se_h += eh[j] * eh[j];
        parity_i ^= (static_cast<int>(qi[j]) & 1);
        parity_h ^= (static_cast<int>(round_even(in[j] - 0.5f)) & 1);
    }
    // The parity correction changes one coordinate, so compare distances
    // after applying the correction (matching the Python reference).
    if (parity_i != 0 || parity_h != 0) {
        if (parity_i != 0) {
            int idx = 0; float best = fabsf(er[0]);
            #pragma unroll
            for (int j = 1; j < 8; ++j) { float a = fabsf(er[j]); if (a > best) { best = a; idx = j; } }
            qi[idx] += (er[idx] >= 0.0f) ? 1.0f : -1.0f;
        }
        if (parity_h != 0) {
            int idx = 0; float best = fabsf(eh[0]);
            #pragma unroll
            for (int j = 1; j < 8; ++j) { float a = fabsf(eh[j]); if (a > best) { best = a; idx = j; } }
            qh[idx] += (eh[idx] >= 0.0f) ? 1.0f : -1.0f;
        }
        se_i = se_h = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) { float di = in[j] - qi[j], dh = in[j] - qh[j]; se_i += di * di; se_h += dh * dh; }
    }
    const bool use_integer = se_i < se_h;
    float* chosen = use_integer ? qi : qh;
    #pragma unroll
    for (int j = 0; j < 8; ++j) dst[j] = chosen[j];
}

torch::Tensor e8_nearest_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    const int64_t n = x.size(0);
    e8_nearest_kernel<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "E8 CUDA kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
