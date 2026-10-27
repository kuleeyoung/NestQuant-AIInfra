#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "nqrt4_cooperative_device.cuh"

template<bool SKIP_FIRST_E8, bool SKIP_SECOND_E8>
__device__ __forceinline__ unsigned nqrt4_decode_breakdown_4lane(
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
    float n0=0.0f,n1=0.0f;
    if(!SKIP_FIRST_E8)nqrt4_nearest_e8_4lane(u0,u1,n0,n1);
    const float s0=coset_at(j0)-(u0-n0),s1=coset_at(j1)-(u1-n1);
    float q0=0.0f,q1=0.0f;
    if(!SKIP_SECOND_E8)nqrt4_nearest_e8_4lane(s0/14.0f,s1/14.0f,q0,q1);
    const int beta_id=(beta_ids[block>>2]>>((block&3)*2))&3;
    const float scale=betas[beta_id]*row_norm*inverse_sqrt_feature;
    return nqrt4_pack_bf16(__float2bfloat16((s0-14.0f*q0)*scale),
                            __float2bfloat16((s1-14.0f*q1)*scale));
}

template<int MODE, bool M1>
__global__ void nqrt4_mma_j2_breakdown_partial(
    const __nv_bfloat16* activation, const uint8_t* codes,
    const uint8_t* beta_ids, const float* row_norms, const float* betas,
    int k_size, int n_size, int64_t tensor_offset, int64_t seed,
    float inverse_sqrt_feature, float* partial) {
    const unsigned lane=threadIdx.x, group=lane>>2, thread=lane&3, k_pair=thread*2;
    const int output_base=int(blockIdx.x)*8, output_channel=output_base+int(group);
    const int split=int(blockIdx.y), split_count=int(gridDim.y), total_tiles=k_size/16;
    const int tile_begin=(total_tiles*split)/split_count;
    const int tile_end=(total_tiles*(split+1))/split_count;
    float c0=0,c1=0,c2=0,c3=0;
    for(int tile=tile_begin; tile<tile_end; tile++) {
        const int ak=tile*16;
        unsigned a0=0,a1=0,a2=0,a3=0;
        if(!M1 || group==0) {
            const int row0=M1?0:int(group), row1=M1?0:int(group+8);
            a0=nqrt4_pack_bf16(activation[row0*k_size+ak+k_pair],activation[row0*k_size+ak+k_pair+1]);
            a2=nqrt4_pack_bf16(activation[row0*k_size+ak+k_pair+8],activation[row0*k_size+ak+k_pair+9]);
            if(!M1) {
                a1=nqrt4_pack_bf16(activation[row1*k_size+ak+k_pair],activation[row1*k_size+ak+k_pair+1]);
                a3=nqrt4_pack_bf16(activation[row1*k_size+ak+k_pair+8],activation[row1*k_size+ak+k_pair+9]);
            }
        }
        const int blocks_per_row=k_size/8;
        const int block=output_channel*blocks_per_row+tile*2;
        unsigned b0,b1;
        if(MODE==0) {
            b0=nqrt4_decode_block_4lane(codes,beta_ids,betas,row_norms[output_channel],block,
                tensor_offset+block,seed,inverse_sqrt_feature);
            b1=nqrt4_decode_block_4lane(codes,beta_ids,betas,row_norms[output_channel],block+1,
                tensor_offset+block+1,seed,inverse_sqrt_feature);
        } else if(MODE==1) {
            // Packed-affine proxy: retain packed code/beta/norm reads, scaling,
            // BF16 conversion and MMA while removing hash and both E8 searches.
            const int j0=thread*2, j1=j0+1;
            auto proxy=[&](int b) {
                auto code_at=[&](int j) { const int index=b*8+j;
                    return float((codes[index>>1]>>((index&1)*4))&15)-7.5f; };
                const int beta_id=(beta_ids[b>>2]>>((b&3)*2))&3;
                const float scale=betas[beta_id]*row_norms[output_channel]*inverse_sqrt_feature;
                return nqrt4_pack_bf16(__float2bfloat16(code_at(j0)*scale),
                                        __float2bfloat16(code_at(j1)*scale));
            };
            b0=proxy(block); b1=proxy(block+1);
        } else if(MODE==2) {
            // MMA floor: activation loads, a constant BF16 B fragment, MMA and stores.
            const __nv_bfloat16 one=__float2bfloat16(1.0f);
            b0=nqrt4_pack_bf16(one,one); b1=b0;
        } else {
            // Modes 3 and 4 remove exactly one E8 nearest-point search while
            // retaining the stateless hash and the rest of the decode path.
            if(MODE==3) {
                b0=nqrt4_decode_breakdown_4lane<true,false>(codes,beta_ids,betas,row_norms[output_channel],block,
                    tensor_offset+block,seed,inverse_sqrt_feature);
                b1=nqrt4_decode_breakdown_4lane<true,false>(codes,beta_ids,betas,row_norms[output_channel],block+1,
                    tensor_offset+block+1,seed,inverse_sqrt_feature);
            } else {
                b0=nqrt4_decode_breakdown_4lane<false,true>(codes,beta_ids,betas,row_norms[output_channel],block,
                    tensor_offset+block,seed,inverse_sqrt_feature);
                b1=nqrt4_decode_breakdown_4lane<false,true>(codes,beta_ids,betas,row_norms[output_channel],block+1,
                    tensor_offset+block+1,seed,inverse_sqrt_feature);
            }
        }
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    const unsigned column=unsigned(output_base)+thread*2;
    if(M1) {
        if(group==0) {
            const int base=split*n_size;
            partial[base+column]=c0; partial[base+column+1]=c1;
        }
    } else {
        const int base=split*16*n_size;
        partial[base+group*n_size+column]=c0; partial[base+group*n_size+column+1]=c1;
        partial[base+(group+8)*n_size+column]=c2; partial[base+(group+8)*n_size+column+1]=c3;
    }
}

torch::Tensor nqrt4_mma_j2_breakdown_cuda(
    torch::Tensor activation, torch::Tensor codes, torch::Tensor beta_ids,
    torch::Tensor row_norms, torch::Tensor betas, int64_t offset, int64_t seed,
    int64_t splits, int64_t mode) {
    TORCH_CHECK(activation.is_cuda() && activation.scalar_type()==torch::kBFloat16 && activation.dim()==2);
    const int64_t m=activation.size(0), k=activation.size(1), n=row_norms.numel();
    TORCH_CHECK((m==1 || m==16) && k>0 && k%16==0 && n>0 && n%8==0);
    TORCH_CHECK(splits>0 && splits<=k/16 && mode>=0 && mode<=4);
    TORCH_CHECK(codes.is_cuda() && beta_ids.is_cuda() && row_norms.is_cuda() && betas.is_cuda());
    auto partial=torch::empty({splits,m,n},activation.options().dtype(torch::kFloat32));
    dim3 grid(unsigned(n/8),unsigned(splits));
    auto launch=[&](auto mode_tag, auto m1_tag) {
        constexpr int kernel_mode=decltype(mode_tag)::value;
        constexpr bool kernel_m1=decltype(m1_tag)::value;
        const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
        nqrt4_mma_j2_breakdown_partial<kernel_mode,kernel_m1><<<grid,32,0,stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),codes.data_ptr<uint8_t>(),
            beta_ids.data_ptr<uint8_t>(),row_norms.data_ptr<float>(),betas.data_ptr<float>(),
            int(k),int(n),offset,seed,rsqrtf(float(k)),partial.data_ptr<float>());
    };
    if(m==1) {
        if(mode==0) launch(std::integral_constant<int,0>{},std::true_type{});
        else if(mode==1) launch(std::integral_constant<int,1>{},std::true_type{});
        else if(mode==2) launch(std::integral_constant<int,2>{},std::true_type{});
        else if(mode==3) launch(std::integral_constant<int,3>{},std::true_type{});
        else launch(std::integral_constant<int,4>{},std::true_type{});
    } else {
        if(mode==0) launch(std::integral_constant<int,0>{},std::false_type{});
        else if(mode==1) launch(std::integral_constant<int,1>{},std::false_type{});
        else if(mode==2) launch(std::integral_constant<int,2>{},std::false_type{});
        else if(mode==3) launch(std::integral_constant<int,3>{},std::false_type{});
        else launch(std::integral_constant<int,4>{},std::false_type{});
    }
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"J2 breakdown launch failed: ",cudaGetErrorString(error));
    return partial;
}
