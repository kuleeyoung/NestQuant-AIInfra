#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void probe(float* c, int source_lane, int source_register, int source_half, int map_a) {
    const unsigned int lane = threadIdx.x;
    const unsigned int one = 0x3f80;
    unsigned int a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    if (source_lane < 0) {
        a0=a1=a2=a3=b0=b1=0x3f803f80;
    } else if (lane == unsigned(source_lane)) {
        unsigned int value = one << (source_half * 16);
        if (map_a) {
            if (source_register == 0) a0 = value;
            if (source_register == 1) a1 = value;
            if (source_register == 2) a2 = value;
            if (source_register == 3) a3 = value;
        } else {
            if (source_register == 0) b0 = value;
            if (source_register == 1) b1 = value;
        }
    }
    if (source_lane >= 0) {
        if (map_a) { b0 = 0x3f803f80; b1 = 0x3f803f80; }
        else { a0 = 0x3f803f80; a1 = 0x3f803f80; a2 = 0x3f803f80; a3 = 0x3f803f80; }
    }
    float c0=0,c1=0,c2=0,c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    const unsigned int row = lane >> 2;
    const unsigned int column = (lane & 3) << 1;
    c[row * 8 + column] = c0;
    c[row * 8 + column + 1] = c1;
    c[(row + 8) * 8 + column] = c2;
    c[(row + 8) * 8 + column + 1] = c3;
}

int main(int argc, char** argv) {
    const int source_lane = argc > 1 ? std::atoi(argv[1]) : -1;
    const int source_register = argc > 2 ? std::atoi(argv[2]) : 0;
    const int source_half = argc > 3 ? std::atoi(argv[3]) : 0;
    const int map_a = argc > 4 ? std::atoi(argv[4]) : 1;
    float *c;
    cudaMalloc(&c, 128 * sizeof(float));
    cudaMemset(c, 0, 128 * sizeof(float));
    probe<<<1,32>>>(c, source_lane, source_register, source_half, map_a);
    auto status = cudaDeviceSynchronize();
    float host[128] = {};
    cudaMemcpy(host, c, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("source=%d,%d,%d map_a=%d status=%d\n", source_lane, source_register, source_half, map_a, int(status));
    for (int row = 0; row < 16; ++row) {
        std::printf("row=%d", row);
        for (int column = 0; column < 8; ++column) std::printf(" %f", host[row*8+column]);
        std::printf("\n");
    }
    cudaFree(c);
    return status == cudaSuccess ? 0 : 1;
}
