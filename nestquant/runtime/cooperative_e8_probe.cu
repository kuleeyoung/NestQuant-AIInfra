#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do {                                                     \
    cudaError_t error = (call);                                                    \
    if (error != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                     cudaGetErrorString(error));                                   \
        std::exit(1);                                                              \
    }                                                                              \
} while (0)

// Four adjacent lanes cooperate on one eight-dimensional E8 vector.  Each lane
// owns two coordinates, so no lane builds the full vector in local memory.
__device__ __forceinline__ void cooperative_nearest_e8_4lane(
    float x0, float x1, float &out0, float &out1) {
    constexpr unsigned kFullMask = 0xffffffffu;
    const int lane_in_group = threadIdx.x & 3;
    const int coord0 = 2 * lane_in_group;

    float integer0 = nearbyintf(x0);
    float integer1 = nearbyintf(x1);
    const float base0 = nearbyintf(x0 - 0.5f);
    const float base1 = nearbyintf(x1 - 0.5f);
    float half0 = base0 + 0.5f;
    float half1 = base1 + 0.5f;
    const float residual_i0 = x0 - integer0;
    const float residual_i1 = x1 - integer1;
    const float residual_h0 = x0 - half0;
    const float residual_h1 = x1 - half1;

    int parity_i = (static_cast<int>(integer0) & 1) ^
                   (static_cast<int>(integer1) & 1);
    int parity_h = (static_cast<int>(base0) & 1) ^
                   (static_cast<int>(base1) & 1);
    parity_i ^= __shfl_xor_sync(kFullMask, parity_i, 1, 4);
    parity_i ^= __shfl_xor_sync(kFullMask, parity_i, 2, 4);
    parity_h ^= __shfl_xor_sync(kFullMask, parity_h, 1, 4);
    parity_h ^= __shfl_xor_sync(kFullMask, parity_h, 2, 4);

    // Encode the tie-break rule explicitly: larger residual wins; equal
    // residuals choose the smaller coordinate, matching the scalar loop.
    float best_i = fabsf(residual_i0);
    int index_i = coord0;
    if (fabsf(residual_i1) > best_i) {
        best_i = fabsf(residual_i1);
        index_i = coord0 + 1;
    }
    float best_h = fabsf(residual_h0);
    int index_h = coord0;
    if (fabsf(residual_h1) > best_h) {
        best_h = fabsf(residual_h1);
        index_h = coord0 + 1;
    }

#pragma unroll
    for (int delta = 1; delta < 4; delta <<= 1) {
        const float other_i = __shfl_xor_sync(kFullMask, best_i, delta, 4);
        const int other_index_i = __shfl_xor_sync(kFullMask, index_i, delta, 4);
        if (other_i > best_i || (other_i == best_i && other_index_i < index_i)) {
            best_i = other_i;
            index_i = other_index_i;
        }
        const float other_h = __shfl_xor_sync(kFullMask, best_h, delta, 4);
        const int other_index_h = __shfl_xor_sync(kFullMask, index_h, delta, 4);
        if (other_h > best_h || (other_h == best_h && other_index_h < index_h)) {
            best_h = other_h;
            index_h = other_index_h;
        }
    }

    if (parity_i) {
        if (index_i == coord0) integer0 += residual_i0 >= 0.0f ? 1.0f : -1.0f;
        if (index_i == coord0 + 1) integer1 += residual_i1 >= 0.0f ? 1.0f : -1.0f;
    }
    if (parity_h) {
        if (index_h == coord0) half0 += residual_h0 >= 0.0f ? 1.0f : -1.0f;
        if (index_h == coord0 + 1) half1 += residual_h1 >= 0.0f ? 1.0f : -1.0f;
    }

    float error_i = (x0 - integer0) * (x0 - integer0) +
                    (x1 - integer1) * (x1 - integer1);
    float error_h = (x0 - half0) * (x0 - half0) +
                    (x1 - half1) * (x1 - half1);
    error_i += __shfl_xor_sync(kFullMask, error_i, 1, 4);
    error_i += __shfl_xor_sync(kFullMask, error_i, 2, 4);
    error_h += __shfl_xor_sync(kFullMask, error_h, 1, 4);
    error_h += __shfl_xor_sync(kFullMask, error_h, 2, 4);
    const bool choose_integer = error_i < error_h;  // Half lattice wins ties.
    out0 = choose_integer ? integer0 : half0;
    out1 = choose_integer ? integer1 : half1;
}

__device__ __forceinline__ void scalar_nearest_e8(const float *x, float *out) {
    float integer[8], half[8], residual_i[8], residual_h[8];
    int parity_i = 0, parity_h = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        integer[j] = nearbyintf(x[j]);
        const float base = nearbyintf(x[j] - 0.5f);
        half[j] = base + 0.5f;
        residual_i[j] = x[j] - integer[j];
        residual_h[j] = x[j] - half[j];
        parity_i ^= static_cast<int>(integer[j]) & 1;
        parity_h ^= static_cast<int>(base) & 1;
    }
    if (parity_i) {
        int index = 0;
        float best = fabsf(residual_i[0]);
#pragma unroll
        for (int j = 1; j < 8; ++j) {
            const float value = fabsf(residual_i[j]);
            if (value > best) { best = value; index = j; }
        }
        integer[index] += residual_i[index] >= 0.0f ? 1.0f : -1.0f;
    }
    if (parity_h) {
        int index = 0;
        float best = fabsf(residual_h[0]);
#pragma unroll
        for (int j = 1; j < 8; ++j) {
            const float value = fabsf(residual_h[j]);
            if (value > best) { best = value; index = j; }
        }
        half[index] += residual_h[index] >= 0.0f ? 1.0f : -1.0f;
    }
    float error_i = 0.0f, error_h = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float di = x[j] - integer[j];
        const float dh = x[j] - half[j];
        error_i += di * di;
        error_h += dh * dh;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = error_i < error_h ? integer[j] : half[j];
}

__global__ void cooperative_probe(const float *input, float *output, int vectors) {
    const int global_lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int vector = global_lane >> 2;
    const int lane_in_group = global_lane & 3;
    if (vector >= vectors) return;
    const int base = vector * 8 + lane_in_group * 2;
    float out0, out1;
    cooperative_nearest_e8_4lane(input[base], input[base + 1], out0, out1);
    output[base] = out0;
    output[base + 1] = out1;
}

__global__ void scalar_probe(const float *input, float *output, int vectors) {
    const int vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector < vectors) scalar_nearest_e8(input + vector * 8, output + vector * 8);
}

int main() {
    constexpr int kVectors = 4096;
    constexpr int kValues = kVectors * 8;
    std::vector<float> input(kValues);
    unsigned state = 0x12345678u;
    for (int i = 0; i < kValues; ++i) {
        state = state * 1664525u + 1013904223u;
        input[i] = (static_cast<int>(state >> 8) % 40001 - 20000) / 997.0f;
    }
    // Exact half/integer boundaries and repeated residuals exercise tie breaks.
    const float boundary[][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
        {-0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f},
        {1.5f, -1.5f, 2.5f, -2.5f, 3.5f, -3.5f, 4.5f, -4.5f},
        {0.49f, -0.49f, 1.49f, -1.49f, 2.49f, -2.49f, 3.49f, -3.49f},
        {0.25f, -0.25f, 1.25f, -1.25f, 2.25f, -2.25f, 3.25f, -3.25f},
    };
    for (int v = 0; v < static_cast<int>(sizeof(boundary) / sizeof(boundary[0])); ++v)
        for (int j = 0; j < 8; ++j) input[v * 8 + j] = boundary[v][j];

    float *device_input, *device_cooperative, *device_scalar;
    CUDA_CHECK(cudaMalloc(&device_input, kValues * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_cooperative, kValues * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_scalar, kValues * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), kValues * sizeof(float), cudaMemcpyHostToDevice));
    cooperative_probe<<<(kVectors * 4 + 127) / 128, 128>>>(device_input, device_cooperative, kVectors);
    scalar_probe<<<(kVectors + 127) / 128, 128>>>(device_input, device_scalar, kVectors);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> cooperative(kValues), scalar(kValues);
    CUDA_CHECK(cudaMemcpy(cooperative.data(), device_cooperative, kValues * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scalar.data(), device_scalar, kValues * sizeof(float), cudaMemcpyDeviceToHost));
    int mismatches = 0;
    float max_error = 0.0f;
    for (int i = 0; i < kValues; ++i) {
        const float error = fabsf(cooperative[i] - scalar[i]);
        max_error = fmaxf(max_error, error);
        if (error != 0.0f && mismatches++ < 8)
            std::printf("mismatch vector=%d coord=%d cooperative=%g scalar=%g\n",
                        i / 8, i % 8, cooperative[i], scalar[i]);
    }
    std::printf("vectors=%d values=%d mismatches=%d max_abs_error=%g\n",
                kVectors, kValues, mismatches, max_error);
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_cooperative));
    CUDA_CHECK(cudaFree(device_scalar));
    return mismatches == 0 ? 0 : 2;
}
