#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdint>

__host__ __device__ void nearest_e8_8(const float* x, float* out) {
    float integer[8], half[8], residual_i[8], residual_h[8];
    int parity_i=0, parity_h=0;
    #pragma unroll
    for(int j=0;j<8;j++) {
        integer[j]=nearbyintf(x[j]);
        float base=nearbyintf(x[j]-0.5f);
        half[j]=base+0.5f;
        residual_i[j]=x[j]-integer[j];
        residual_h[j]=x[j]-half[j];
        parity_i ^= (int(integer[j]) & 1);
        parity_h ^= (int(base) & 1);
    }
    if(parity_i) {
        int idx=0; float best=fabsf(residual_i[0]);
        #pragma unroll
        for(int j=1;j<8;j++) { float v=fabsf(residual_i[j]); if(v>best){best=v;idx=j;} }
        integer[idx] += residual_i[idx]>=0.0f ? 1.0f : -1.0f;
    }
    if(parity_h) {
        int idx=0; float best=fabsf(residual_h[0]);
        #pragma unroll
        for(int j=1;j<8;j++) { float v=fabsf(residual_h[j]); if(v>best){best=v;idx=j;} }
        half[idx] += residual_h[idx]>=0.0f ? 1.0f : -1.0f;
    }
    float error_i=0.0f,error_h=0.0f;
    #pragma unroll
    for(int j=0;j<8;j++) {
        float di=x[j]-integer[j], dh=x[j]-half[j];
        error_i += di*di; error_h += dh*dh;
    }
    const bool use_integer=error_i<error_h;
    #pragma unroll
    for(int j=0;j<8;j++) out[j]=use_integer?integer[j]:half[j];
}

__host__ __device__ float stateless_uniform(int64_t counter, int64_t seed) {
    int64_t random=(counter+seed)&0x7fffffffLL;
    random=random^(random>>16);
    random=(random*1103515245LL+12345LL)&0x7fffffffLL;
    random=random^(random>>13);
    random=(random*134775813LL+1LL)&0x7fffffffLL;
    random=random^(random>>16);
    return float(random)*(2.0f/2147483648.0f);
}

__global__ void generate_dither(float* output, float* reconstructed, __nv_bfloat16* scaled_bf16, int64_t block_offset, int64_t seed) {
    const int block=threadIdx.x;
    if(block>=16) return;
    float uniform[8], nearest[8], codes[8], coset[8], shifted[8], scaled[8], coarse[8];
    #pragma unroll
    for(int j=0;j<8;j++) uniform[j]=stateless_uniform((block_offset+block)*8+j,seed);
    nearest_e8_8(uniform,nearest);
    #pragma unroll
    for(int j=0;j<8;j++) {
        output[block*8+j]=uniform[j]-nearest[j];
        codes[j]=float(((block*8+j)*5+3)%14);
    }
    const float half_c7=0.5f*codes[7];
    coset[0]=2.0f*codes[0]-codes[1]+half_c7;
    #pragma unroll
    for(int j=1;j<6;j++) coset[j]=codes[j]-codes[j+1]+half_c7;
    coset[6]=codes[6]+half_c7; coset[7]=half_c7;
    #pragma unroll
    for(int j=0;j<8;j++) { shifted[j]=coset[j]-output[block*8+j]; scaled[j]=shifted[j]/14.0f; }
    nearest_e8_8(scaled,coarse);
    const float betas[4]={0.25f,0.5f,0.75f,1.0f};
    const int beta_id=(block*3+1)&3;
    const int output_row=block>>1;
    const float row_norm=1.0f+0.125f*float(output_row);
    const float scale=betas[beta_id]*row_norm*(1.0f/4.0f); // sqrt(feature_count=16)
    #pragma unroll
    for(int j=0;j<8;j++) {
        reconstructed[block*8+j]=shifted[j]-14.0f*coarse[j];
        scaled_bf16[block*8+j]=__float2bfloat16(reconstructed[block*8+j]*scale);
    }
}

int main() {
    constexpr int64_t offset=37, seed=58;
    float *device,*device_reconstructed,actual[128],reference[128],actual_reconstructed[128],reference_reconstructed[128]; __nv_bfloat16 *device_scaled,actual_scaled[128],reference_scaled[128];
    cudaMalloc(&device,sizeof(actual));
    cudaMalloc(&device_reconstructed,sizeof(actual_reconstructed));
    cudaMalloc(&device_scaled,sizeof(actual_scaled));
    generate_dither<<<1,16>>>(device,device_reconstructed,device_scaled,offset,seed);
    auto status=cudaDeviceSynchronize();
    cudaMemcpy(actual,device,sizeof(actual),cudaMemcpyDeviceToHost);
    cudaMemcpy(actual_reconstructed,device_reconstructed,sizeof(actual_reconstructed),cudaMemcpyDeviceToHost);
    cudaMemcpy(actual_scaled,device_scaled,sizeof(actual_scaled),cudaMemcpyDeviceToHost);
    for(int block=0;block<16;block++) {
        float uniform[8],nearest[8];
        for(int j=0;j<8;j++) uniform[j]=stateless_uniform((offset+block)*8+j,seed);
        nearest_e8_8(uniform,nearest);
        float codes[8],coset[8],shifted[8],scaled[8],coarse[8];
        for(int j=0;j<8;j++) { reference[block*8+j]=uniform[j]-nearest[j]; codes[j]=float(((block*8+j)*5+3)%14); }
        float h=0.5f*codes[7]; coset[0]=2.0f*codes[0]-codes[1]+h;
        for(int j=1;j<6;j++) coset[j]=codes[j]-codes[j+1]+h;
        coset[6]=codes[6]+h; coset[7]=h;
        for(int j=0;j<8;j++) { shifted[j]=coset[j]-reference[block*8+j]; scaled[j]=shifted[j]/14.0f; }
        nearest_e8_8(scaled,coarse);
        const float betas[4]={0.25f,0.5f,0.75f,1.0f}; int beta_id=(block*3+1)&3; int row=block>>1;
        float scale=betas[beta_id]*(1.0f+0.125f*float(row))*0.25f;
        for(int j=0;j<8;j++) { reference_reconstructed[block*8+j]=shifted[j]-14.0f*coarse[j]; reference_scaled[block*8+j]=__float2bfloat16(reference_reconstructed[block*8+j]*scale); }
    }
    int mismatches=0,reconstructed_mismatches=0,bf16_mismatches=0; float max_error=0.0f,max_reconstructed_error=0.0f;
    for(int i=0;i<128;i++) { float e=fabsf(actual[i]-reference[i]); if(e!=0.0f)mismatches++; max_error=fmaxf(max_error,e); float r=fabsf(actual_reconstructed[i]-reference_reconstructed[i]); if(r!=0.0f)reconstructed_mismatches++; max_reconstructed_error=fmaxf(max_reconstructed_error,r); bf16_mismatches += __bfloat16_as_ushort(actual_scaled[i]) != __bfloat16_as_ushort(reference_scaled[i]); }
    std::printf("status=%d dither_mismatches=%d reconstructed_mismatches=%d bf16_mismatches=%d dither_max_error=%g reconstructed_max_error=%g scaled_first=%g\n",
                int(status),mismatches,reconstructed_mismatches,bf16_mismatches,max_error,max_reconstructed_error,__bfloat162float(actual_scaled[0]));
    cudaFree(device); cudaFree(device_reconstructed); cudaFree(device_scaled);
    return status==cudaSuccess && mismatches==0 && reconstructed_mismatches==0 && bf16_mismatches==0 ? 0 : 1;
}
