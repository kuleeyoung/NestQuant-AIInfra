#include <cstdio>

#include <cuda_runtime.h>
#include <cutlass/version.h>
#include <cute/tensor.hpp>

__global__ void cute_layout_probe(int* output) {
    using Layout = decltype(cute::make_layout(
        cute::make_shape(cute::Int<16>{}, cute::Int<16>{})
    ));
    static_assert(cute::size(Layout{}) == 256);
    if (threadIdx.x == 0) {
        output[0] = int(cute::size(Layout{}));
    }
}

int main() {
    int* device_output = nullptr;
    int host_output = 0;
    if (cudaMalloc(&device_output, sizeof(int)) != cudaSuccess) {
        return 2;
    }
    cute_layout_probe<<<1, 32>>>(device_output);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(device_output);
        return 3;
    }
    if (cudaMemcpy(
            &host_output, device_output, sizeof(int), cudaMemcpyDeviceToHost
        ) != cudaSuccess) {
        cudaFree(device_output);
        return 4;
    }
    cudaFree(device_output);
    std::printf(
        "CUTLASS %d.%d.%d; CuTe layout elements=%d\n",
        CUTLASS_MAJOR, CUTLASS_MINOR, CUTLASS_PATCH, host_output
    );
    return host_output == 256 ? 0 : 5;
}
