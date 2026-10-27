#pragma once

#include "nqrt4_device.cuh"

__device__ __forceinline__ float nqrt4_fadd_rn(float left,float right){
    float output;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(output) : "f"(left),"f"(right));
    return output;
}

__device__ __forceinline__ float nqrt4_sum8_torch_order_4lane(float value0,float value1){
    // Match PyTorch CUDA's contiguous length-8 reduction order, which is also
#ifdef NQRT4_SUM8_BROADCAST_REFERENCE
    // Diagnostic reference matching the original eight-broadcast implementation.
    const float v0=__shfl_sync(0xffffffffu,value0,0,4);
    const float v1=__shfl_sync(0xffffffffu,value1,0,4);
    const float v2=__shfl_sync(0xffffffffu,value0,1,4);
    const float v3=__shfl_sync(0xffffffffu,value1,1,4);
    const float v4=__shfl_sync(0xffffffffu,value0,2,4);
    const float v5=__shfl_sync(0xffffffffu,value1,2,4);
    const float v6=__shfl_sync(0xffffffffu,value0,3,4);
    const float v7=__shfl_sync(0xffffffffu,value1,3,4);
    const float left=nqrt4_fadd_rn(nqrt4_fadd_rn(v0,v4),nqrt4_fadd_rn(v2,v6));
    const float right=nqrt4_fadd_rn(nqrt4_fadd_rn(v1,v5),nqrt4_fadd_rn(v3,v7));
#else
    // used by the project's Triton reference.  XOR exchanges preserve the
    // exact grouping while halving shuffles from eight broadcasts to four.
    const float even_pair=nqrt4_fadd_rn(
        value0,__shfl_xor_sync(0xffffffffu,value0,2,4));
    const float odd_pair=nqrt4_fadd_rn(
        value1,__shfl_xor_sync(0xffffffffu,value1,2,4));
    const float left=nqrt4_fadd_rn(
        even_pair,__shfl_xor_sync(0xffffffffu,even_pair,1,4));
    const float right=nqrt4_fadd_rn(
        odd_pair,__shfl_xor_sync(0xffffffffu,odd_pair,1,4));
#endif
    return nqrt4_fadd_rn(left,right);
}

__device__ __forceinline__ void nqrt4_nearest_e8_4lane(
    float x0, float x1, float& out0, float& out1) {
    const int lane = threadIdx.x & 3;
    const int coord0 = lane * 2;
    float i0 = nearbyintf(x0), i1 = nearbyintf(x1);
    const float b0 = nearbyintf(x0 - 0.5f), b1 = nearbyintf(x1 - 0.5f);
    float h0 = b0 + 0.5f, h1 = b1 + 0.5f;
    const float ri0 = x0-i0, ri1 = x1-i1, rh0 = x0-h0, rh1 = x1-h1;
    int pi=(int(i0)&1)^(int(i1)&1), ph=(int(b0)&1)^(int(b1)&1);
#ifdef NQRT4_PARITY_SEPARATE_REFERENCE
    pi^=__shfl_xor_sync(0xffffffffu,pi,1,4); pi^=__shfl_xor_sync(0xffffffffu,pi,2,4);
    ph^=__shfl_xor_sync(0xffffffffu,ph,1,4); ph^=__shfl_xor_sync(0xffffffffu,ph,2,4);
#else
    // Reduce the integer- and half-lattice parity bits together. XOR is
    // component-wise over the packed bits, so this is exactly equivalent to
    // four independent shuffle reductions but uses only two shuffle steps.
    int parity=pi|(ph<<1);
    parity^=__shfl_xor_sync(0xffffffffu,parity,1,4);
    parity^=__shfl_xor_sync(0xffffffffu,parity,2,4);
    pi=parity&1; ph=(parity>>1)&1;
#endif
    float vi=fabsf(ri0); int ii=coord0;
    if(fabsf(ri1)>vi){vi=fabsf(ri1);ii=coord0+1;}
    float vh=fabsf(rh0); int ih=coord0;
    if(fabsf(rh1)>vh){vh=fabsf(rh1);ih=coord0+1;}
#ifdef NQRT4_MAX_INDEX_SHUFFLE_REFERENCE
    #pragma unroll
    for(int d=1;d<4;d<<=1){
        float ov=__shfl_xor_sync(0xffffffffu,vi,d,4); int oi=__shfl_xor_sync(0xffffffffu,ii,d,4);
        if(ov>vi||(ov==vi&&oi<ii)){vi=ov;ii=oi;}
        ov=__shfl_xor_sync(0xffffffffu,vh,d,4); oi=__shfl_xor_sync(0xffffffffu,ih,d,4);
        if(ov>vh||(ov==vh&&oi<ih)){vh=ov;ih=oi;}
    }
    if(pi){if(ii==coord0)i0+=ri0>=0?1:-1;if(ii==coord0+1)i1+=ri1>=0?1:-1;}
    if(ph){if(ih==coord0)h0+=rh0>=0?1:-1;if(ih==coord0+1)h1+=rh1>=0?1:-1;}
#else
    const float local_vi=vi, local_vh=vh;
    #pragma unroll
    for(int d=1;d<4;d<<=1){
        vi=fmaxf(vi,__shfl_xor_sync(0xffffffffu,vi,d,4));
        vh=fmaxf(vh,__shfl_xor_sync(0xffffffffu,vh,d,4));
    }
    const int group_base=int(threadIdx.x)&~3;
    const unsigned integer_ties=(__ballot_sync(0xffffffffu,local_vi==vi)>>group_base)&15u;
    const unsigned half_ties=(__ballot_sync(0xffffffffu,local_vh==vh)>>group_base)&15u;
    const int integer_lane=__ffs(int(integer_ties))-1;
    const int half_lane=__ffs(int(half_ties))-1;
    if(pi&&lane==integer_lane){if(ii==coord0)i0+=ri0>=0?1:-1;if(ii==coord0+1)i1+=ri1>=0?1:-1;}
    if(ph&&lane==half_lane){if(ih==coord0)h0+=rh0>=0?1:-1;if(ih==coord0+1)h1+=rh1>=0?1:-1;}
#endif
    const float di0=x0-i0,di1=x1-i1,dh0=x0-h0,dh1=x1-h1;
    const float ei=nqrt4_sum8_torch_order_4lane(di0*di0,di1*di1);
    const float eh=nqrt4_sum8_torch_order_4lane(dh0*dh0,dh1*dh1);
    out0=ei<eh?i0:h0;out1=ei<eh?i1:h1;
}

__device__ __forceinline__ unsigned nqrt4_decode_block_4lane(
    const uint8_t* codes,const uint8_t* beta_ids,const float* betas,float row_norm,
    int block,int64_t global_block,int64_t seed,float inverse_sqrt_feature) {
    const int lane=threadIdx.x&3, j0=lane*2, j1=j0+1;
    auto code_at=[&](int j){const int index=block*8+j;return float((codes[index>>1]>>((index&1)*4))&15);};
    const float c7=code_at(7), half=0.5f*c7;
    auto coset_at=[&](int j){
        if(j==0)return 2.0f*code_at(0)-code_at(1)+half;
        if(j<6)return code_at(j)-code_at(j+1)+half;
        if(j==6)return code_at(6)+half;
        return half;
    };
    const float u0=nqrt4_uniform(global_block*8+j0,seed);
    const float u1=nqrt4_uniform(global_block*8+j1,seed);
    float n0,n1;nqrt4_nearest_e8_4lane(u0,u1,n0,n1);
    const float s0=coset_at(j0)-(u0-n0),s1=coset_at(j1)-(u1-n1);
    float q0,q1;nqrt4_nearest_e8_4lane(s0/14.0f,s1/14.0f,q0,q1);
    const int beta_id=(beta_ids[block>>2]>>((block&3)*2))&3;
    const float scale=betas[beta_id]*row_norm*inverse_sqrt_feature;
    return nqrt4_pack_bf16(__float2bfloat16((s0-14.0f*q0)*scale),
                            __float2bfloat16((s1-14.0f*q1)*scale));
}
