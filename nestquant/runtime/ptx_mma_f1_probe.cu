#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

__device__ __forceinline__ unsigned pack2(__nv_bfloat16 x, __nv_bfloat16 y) {
    return unsigned(__bfloat16_as_ushort(x)) |
           (unsigned(__bfloat16_as_ushort(y)) << 16);
}

__global__ void mma_f1(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c) {
    const unsigned lane = threadIdx.x;
    const unsigned group = lane >> 2;
    const unsigned thread = lane & 3;
    const unsigned k0 = thread * 2;
    const unsigned column = group;

    unsigned a0 = pack2(a[group * 16 + k0], a[group * 16 + k0 + 1]);
    unsigned a1 = pack2(a[(group + 8) * 16 + k0], a[(group + 8) * 16 + k0 + 1]);
    unsigned a2 = pack2(a[group * 16 + k0 + 8], a[group * 16 + k0 + 9]);
    unsigned a3 = pack2(a[(group + 8) * 16 + k0 + 8], a[(group + 8) * 16 + k0 + 9]);
    unsigned b0 = pack2(b[k0 * 8 + column], b[(k0 + 1) * 8 + column]);
    unsigned b1 = pack2(b[(k0 + 8) * 8 + column], b[(k0 + 9) * 8 + column]);
    float c0=0,c1=0,c2=0,c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    const unsigned row = group;
    const unsigned col = thread * 2;
    c[row*8+col]=c0; c[row*8+col+1]=c1;
    c[(row+8)*8+col]=c2; c[(row+8)*8+col+1]=c3;
}

int main() {
    __nv_bfloat16 ha[256], hb[128];
    float reference[128] = {}, result[128] = {};
    for (int i=0;i<16;i++) for (int k=0;k<16;k++)
        ha[i*16+k] = __float2bfloat16(float((i*3+k*5)%17-8)/4.0f);
    for (int k=0;k<16;k++) for (int j=0;j<8;j++)
        hb[k*8+j] = __float2bfloat16(float((k*7+j*2)%13-6)/4.0f);
    for (int i=0;i<16;i++) for (int j=0;j<8;j++) for (int k=0;k<16;k++)
        reference[i*8+j] += __bfloat162float(ha[i*16+k]) * __bfloat162float(hb[k*8+j]);
    __nv_bfloat16 *da,*db; float *dc;
    cudaMalloc(&da,sizeof(ha)); cudaMalloc(&db,sizeof(hb)); cudaMalloc(&dc,sizeof(result));
    cudaMemcpy(da,ha,sizeof(ha),cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,sizeof(hb),cudaMemcpyHostToDevice);
    mma_f1<<<1,32>>>(da,db,dc);
    auto status=cudaDeviceSynchronize();
    cudaMemcpy(result,dc,sizeof(result),cudaMemcpyDeviceToHost);
    float max_error=0.0f;
    for(int i=0;i<128;i++) max_error=fmaxf(max_error,fabsf(result[i]-reference[i]));
    std::printf("status=%d max_abs_error=%g first=%g reference_first=%g\n",int(status),max_error,result[0],reference[0]);
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return status==cudaSuccess && max_error < 1e-5f ? 0 : 1;
}
