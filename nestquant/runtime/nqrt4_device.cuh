#pragma once

#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

__device__ __forceinline__ int64_t nqrt4_hash(int64_t counter, int64_t seed) {
    // Every stage is reduced modulo 2^31.  Unsigned 32-bit multiplication has
    // the same low 31 bits as the former int64 path and is substantially
    // cheaper on NVIDIA GPUs.
    uint32_t value=uint32_t(uint64_t(counter)+uint64_t(seed))&0x7fffffffu;
    value ^= value>>16;
    value=(value*1103515245u+12345u)&0x7fffffffu;
    value ^= value>>13;
    value=(value*134775813u+1u)&0x7fffffffu;
    return value^(value>>16);
}

__device__ __forceinline__ float nqrt4_uniform(int64_t counter, int64_t seed) {
    return float(nqrt4_hash(counter,seed))*(2.0f/2147483648.0f);
}

__device__ __forceinline__ void nqrt4_nearest_e8(const float* x, float* out) {
    float integer[8],half[8],ri[8],rh[8]; int pi=0,ph=0;
    #pragma unroll
    for(int j=0;j<8;j++) {
        integer[j]=nearbyintf(x[j]); float base=nearbyintf(x[j]-0.5f);
        half[j]=base+0.5f; ri[j]=x[j]-integer[j]; rh[j]=x[j]-half[j];
        pi^=(int(integer[j])&1); ph^=(int(base)&1);
    }
    if(pi) { int idx=0; float best=fabsf(ri[0]);
        #pragma unroll
        for(int j=1;j<8;j++){float v=fabsf(ri[j]);if(v>best){best=v;idx=j;}}
        integer[idx]+=ri[idx]>=0.0f?1.0f:-1.0f; }
    if(ph) { int idx=0; float best=fabsf(rh[0]);
        #pragma unroll
        for(int j=1;j<8;j++){float v=fabsf(rh[j]);if(v>best){best=v;idx=j;}}
        half[idx]+=rh[idx]>=0.0f?1.0f:-1.0f; }
    float ei=0.0f,eh=0.0f;
    #pragma unroll
    for(int j=0;j<8;j++){float di=x[j]-integer[j],dh=x[j]-half[j];ei+=di*di;eh+=dh*dh;}
    #pragma unroll
    for(int j=0;j<8;j++) out[j]=ei<eh?integer[j]:half[j];
}

__device__ __forceinline__ unsigned nqrt4_pack_bf16(__nv_bfloat16 x, __nv_bfloat16 y) {
    return unsigned(__bfloat16_as_ushort(x))|(unsigned(__bfloat16_as_ushort(y))<<16);
}

__device__ __forceinline__ void nqrt4_decode_block(
    const uint8_t* codes, const uint8_t* beta_ids, const float* betas,
    float row_norm, int block, int64_t global_block, int64_t seed,
    float inverse_sqrt_feature, __nv_bfloat16* output) {
    float code[8], coset[8], uniform[8], nearest[8], shifted[8], scaled[8], coarse[8];
    #pragma unroll
    for(int j=0;j<8;j++) {
        const int index=block*8+j; const uint8_t byte=codes[index>>1];
        code[j]=float((byte>>((index&1)*4))&15);
        uniform[j]=nqrt4_uniform(global_block*8+j,seed);
    }
    nqrt4_nearest_e8(uniform,nearest);
    float h=0.5f*code[7]; coset[0]=2.0f*code[0]-code[1]+h;
    #pragma unroll
    for(int j=1;j<6;j++) coset[j]=code[j]-code[j+1]+h;
    coset[6]=code[6]+h; coset[7]=h;
    #pragma unroll
    for(int j=0;j<8;j++){shifted[j]=coset[j]-(uniform[j]-nearest[j]);scaled[j]=shifted[j]/14.0f;}
    nqrt4_nearest_e8(scaled,coarse);
    const uint8_t beta_byte=beta_ids[block>>2];
    const int beta_id=(beta_byte>>((block&3)*2))&3;
    const float scale=betas[beta_id]*row_norm*inverse_sqrt_feature;
    #pragma unroll
    for(int j=0;j<8;j++) output[j]=__float2bfloat16((shifted[j]-14.0f*coarse[j])*scale);
}
