#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "nqrt4_cooperative_device.cuh"

namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name) {
    TORCH_CHECK(
        tensor.device() == reference.device(), name,
        " must be on the same CUDA device as the code payload");
}

}  // namespace

__global__ void nqa4_inspect_m16_kernel(
    const uint8_t* codes,int block_count,float* output) {
    const int block=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    if(block>=block_count)return;
    auto code_at=[&](int j){const int index=block*8+j;return float((codes[index>>1]>>((index&1)*4))&15);};
    const float c7=code_at(7),half=0.5f*c7;
    auto coset_at=[&](int j){
        if(j==0)return 2.0f*code_at(0)-code_at(1)+half;
        if(j<6)return code_at(j)-code_at(j+1)+half;
        if(j==6)return code_at(6)+half;
        return half;
    };
    float integer[8],half_base[8],half_integer[8],x[8];
    int integer_parity=0,half_parity=0,integer_index=0,half_index=0;
    #pragma unroll
    for(int j=0;j<8;j++){
        x[j]=nqrt4_fmul_rn(coset_at(j),1.0f/14.0f);
        integer[j]=nearbyintf(x[j]);
        half_base[j]=nearbyintf(nqrt4_fadd_rn(x[j],-0.5f));
        half_integer[j]=half_base[j]+0.5f;
        integer_parity^=(int(integer[j])&1);
        half_parity^=(int(half_base[j])&1);
    }
    float integer_max=fabsf(nqrt4_fadd_rn(x[0],-integer[0]));
    float half_max=fabsf(nqrt4_fadd_rn(nqrt4_fadd_rn(x[0],-0.5f),-half_base[0]));
    #pragma unroll
    for(int j=1;j<8;j++){
        const float ri=fabsf(nqrt4_fadd_rn(x[j],-integer[j]));
        const float rh=fabsf(nqrt4_fadd_rn(nqrt4_fadd_rn(x[j],-0.5f),-half_base[j]));
        if(ri>integer_max){integer_max=ri;integer_index=j;}
        if(rh>half_max){half_max=rh;half_index=j;}
    }
    float* row=output+block*26;
    #pragma unroll
    for(int j=0;j<8;j++){
        row[j]=x[j];
        row[8+j]=fabsf(nqrt4_fadd_rn(x[j],-integer[j]));
        row[16+j]=fabsf(nqrt4_fadd_rn(nqrt4_fadd_rn(x[j],-0.5f),-half_base[j]));
    }
    row[24]=float(integer_index+(integer_parity?16:0));
    row[25]=float(half_index+(half_parity?16:0));
}

torch::Tensor nqa4_inspect_m16_cuda(torch::Tensor codes,int64_t k_size) {
    TORCH_CHECK(k_size>0&&k_size%16==0&&k_size<=INT_MAX);
    check_cuda_contiguous(codes,"codes");
    TORCH_CHECK(codes.scalar_type()==torch::kUInt8&&codes.numel()==16*k_size/2);
    c10::cuda::CUDAGuard device_guard(codes.device());
    const int blocks=int(16*k_size/8);
    auto output=torch::empty({blocks,26},codes.options().dtype(torch::kFloat32));
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    nqa4_inspect_m16_kernel<<<unsigned((blocks+255)/256),256,0,stream>>>(
        codes.data_ptr<uint8_t>(),blocks,output.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"A4 inspect launch failed: ",cudaGetErrorString(error));
    return output;
}

__global__ void nqa4_decode_m16_kernel(
    const uint8_t* codes,const uint8_t* beta_ids,const float* row_norms,
    const float* betas,int blocks_per_row,float inverse_sqrt_feature,
    __nv_bfloat16* output) {
    const int subgroup=int(threadIdx.x)>>2;
    const int flat_block=int(blockIdx.x)*8+subgroup;
    const int row=flat_block/blocks_per_row;
    const unsigned pair=nqa4_decode_block_4lane(
        codes,beta_ids,betas,row_norms[row],flat_block,inverse_sqrt_feature);
    reinterpret_cast<unsigned*>(output+flat_block*8)[threadIdx.x&3]=pair;
}

torch::Tensor nqa4_decode_m16_cuda(
    torch::Tensor codes,torch::Tensor beta_ids,torch::Tensor row_norms,
    torch::Tensor betas,int64_t k_size) {
    TORCH_CHECK(k_size>0&&k_size%16==0&&k_size<=INT_MAX);
    check_cuda_contiguous(codes,"codes");
    check_cuda_contiguous(beta_ids,"beta_ids");
    check_cuda_contiguous(row_norms,"row_norms");
    check_cuda_contiguous(betas,"betas");
    check_same_device(codes,beta_ids,"beta_ids");
    check_same_device(codes,row_norms,"row_norms");
    check_same_device(codes,betas,"betas");
    TORCH_CHECK(row_norms.scalar_type()==torch::kFloat32&&row_norms.numel()==16);
    TORCH_CHECK(codes.scalar_type()==torch::kUInt8&&codes.numel()==16*k_size/2);
    TORCH_CHECK(beta_ids.scalar_type()==torch::kUInt8&&beta_ids.numel()==16*k_size/32);
    TORCH_CHECK(betas.scalar_type()==torch::kFloat32&&betas.numel()==4);
    c10::cuda::CUDAGuard device_guard(codes.device());
    auto output=torch::empty({16,k_size},row_norms.options().dtype(torch::kBFloat16));
    const int blocks=int(16*k_size/8);
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(blocks%8==0,"A4 diagnostic block count must be warp aligned");
    nqa4_decode_m16_kernel<<<blocks/8,32,0,stream>>>(
        codes.data_ptr<uint8_t>(),beta_ids.data_ptr<uint8_t>(),row_norms.data_ptr<float>(),
        betas.data_ptr<float>(),int(k_size/8),rsqrtf(float(k_size)),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()));
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"A4 decode launch failed: ",cudaGetErrorString(error));
    return output;
}

__global__ void nqrt4_w4a4_mma_m16_kernel(
    const uint8_t* act_codes,const uint8_t* act_beta_ids,
    const float* act_row_norms,const float* act_betas,
    const uint8_t* weight_codes,const uint8_t* weight_beta_ids,
    const float* weight_row_norms,const float* weight_betas,
    int k_size,int n_size,int64_t weight_offset,int64_t weight_seed,
    float inverse_sqrt_feature,float* output) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3;
    const int output_base=int(blockIdx.x)*8;
    const int output_channel=output_base+int(group);
    const int blocks_per_row=k_size/8;
    float c0=0,c1=0,c2=0,c3=0;
    for(int tile=0;tile<k_size/16;tile++){
        const int row0=int(group),row1=row0+8;
        const int act_block0=row0*blocks_per_row+tile*2;
        const int act_block1=row1*blocks_per_row+tile*2;
        const unsigned a0=nqa4_decode_block_4lane(
            act_codes,act_beta_ids,act_betas,act_row_norms[row0],act_block0,
            inverse_sqrt_feature);
        const unsigned a1=nqa4_decode_block_4lane(
            act_codes,act_beta_ids,act_betas,act_row_norms[row1],act_block1,
            inverse_sqrt_feature);
        const unsigned a2=nqa4_decode_block_4lane(
            act_codes,act_beta_ids,act_betas,act_row_norms[row0],act_block0+1,
            inverse_sqrt_feature);
        const unsigned a3=nqa4_decode_block_4lane(
            act_codes,act_beta_ids,act_betas,act_row_norms[row1],act_block1+1,
            inverse_sqrt_feature);
        const int weight_block=output_channel*blocks_per_row+tile*2;
        const unsigned b0=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output_channel],weight_block,
            weight_offset+weight_block,weight_seed,inverse_sqrt_feature);
        const unsigned b1=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output_channel],weight_block+1,
            weight_offset+weight_block+1,weight_seed,inverse_sqrt_feature);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    const unsigned column=unsigned(output_base)+thread*2;
    output[group*n_size+column]=c0;
    output[group*n_size+column+1]=c1;
    output[(group+8)*n_size+column]=c2;
    output[(group+8)*n_size+column+1]=c3;
}

torch::Tensor nqrt4_w4a4_mma_m16_cuda(
    torch::Tensor act_codes,torch::Tensor act_beta_ids,
    torch::Tensor act_row_norms,torch::Tensor act_betas,
    torch::Tensor weight_codes,torch::Tensor weight_beta_ids,
    torch::Tensor weight_row_norms,torch::Tensor weight_betas,
    int64_t weight_offset,int64_t weight_seed) {
    check_cuda_contiguous(act_codes,"act_codes");
    check_cuda_contiguous(act_beta_ids,"act_beta_ids");
    check_cuda_contiguous(act_row_norms,"act_row_norms");
    check_cuda_contiguous(act_betas,"act_betas");
    check_cuda_contiguous(weight_codes,"weight_codes");
    check_cuda_contiguous(weight_beta_ids,"weight_beta_ids");
    check_cuda_contiguous(weight_row_norms,"weight_row_norms");
    check_cuda_contiguous(weight_betas,"weight_betas");
    check_same_device(act_codes,act_beta_ids,"act_beta_ids");
    check_same_device(act_codes,act_row_norms,"act_row_norms");
    check_same_device(act_codes,act_betas,"act_betas");
    check_same_device(act_codes,weight_codes,"weight_codes");
    check_same_device(act_codes,weight_beta_ids,"weight_beta_ids");
    check_same_device(act_codes,weight_row_norms,"weight_row_norms");
    check_same_device(act_codes,weight_betas,"weight_betas");
    TORCH_CHECK(act_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_row_norms.scalar_type()==torch::kFloat32&&act_row_norms.numel()==16);
    TORCH_CHECK(act_betas.scalar_type()==torch::kFloat32&&act_betas.numel()==4);
    TORCH_CHECK(weight_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(weight_betas.scalar_type()==torch::kFloat32&&weight_betas.numel()==4);
    c10::cuda::CUDAGuard device_guard(act_codes.device());
    const int64_t n_size=weight_row_norms.numel();
    TORCH_CHECK(n_size>0&&n_size%8==0,"N must be a positive multiple of 8");
    TORCH_CHECK(weight_codes.numel()%n_size==0,"weight payload must divide by N");
    const int64_t k_size=weight_codes.numel()*2/n_size;
    TORCH_CHECK(k_size>0&&k_size%16==0&&k_size<=INT_MAX,"K must be a supported multiple of 16");
    TORCH_CHECK(act_codes.numel()==16*k_size/2,"A4 code payload shape mismatch");
    TORCH_CHECK(act_beta_ids.numel()==16*k_size/32,"A4 beta payload shape mismatch");
    TORCH_CHECK(weight_beta_ids.numel()==n_size*k_size/32,"W4 beta payload shape mismatch");
    TORCH_CHECK(n_size<=INT_MAX);
    auto output=torch::empty({16,n_size},weight_row_norms.options());
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    nqrt4_w4a4_mma_m16_kernel<<<unsigned(n_size/8),32,0,stream>>>(
        act_codes.data_ptr<uint8_t>(),act_beta_ids.data_ptr<uint8_t>(),
        act_row_norms.data_ptr<float>(),act_betas.data_ptr<float>(),
        weight_codes.data_ptr<uint8_t>(),weight_beta_ids.data_ptr<uint8_t>(),
        weight_row_norms.data_ptr<float>(),weight_betas.data_ptr<float>(),
        int(k_size),int(n_size),weight_offset,weight_seed,rsqrtf(float(k_size)),
        output.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 MMA launch failed: ",cudaGetErrorString(error));
    return output;
}

// TinyGEMM-style M8 probe.  Compute W[N,K] x X^T[K,8] so the eight token
// rows occupy the native N=8 side of m16n8k16 while sixteen output channels
// occupy its M=16 side.  The result is stored transposed back as [8,N].
__global__ void nqrt4_w4a4_mma_small_m_kernel(
    const uint8_t* act_codes,const uint8_t* act_beta_ids,
    const float* act_row_norms,const float* act_betas,
    const uint8_t* weight_codes,const uint8_t* weight_beta_ids,
    const float* weight_row_norms,const float* weight_betas,
    int m_size,int k_size,int n_size,int64_t weight_offset,int64_t weight_seed,
    float inverse_sqrt_feature,float* output) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3;
    const int output_base=int(blockIdx.x)*16;
    const int output0=output_base+int(group),output1=output0+8;
    const int blocks_per_row=k_size/8;
    float c0=0,c1=0,c2=0,c3=0;
    for(int tile=0;tile<k_size/16;tile++){
        const int weight_block0=output0*blocks_per_row+tile*2;
        const int weight_block1=output1*blocks_per_row+tile*2;
        const unsigned a0=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output0],weight_block0,
            weight_offset+weight_block0,weight_seed,inverse_sqrt_feature);
        const unsigned a1=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output1],weight_block1,
            weight_offset+weight_block1,weight_seed,inverse_sqrt_feature);
        const unsigned a2=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output0],weight_block0+1,
            weight_offset+weight_block0+1,weight_seed,inverse_sqrt_feature);
        const unsigned a3=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output1],weight_block1+1,
            weight_offset+weight_block1+1,weight_seed,inverse_sqrt_feature);
        const bool active=int(group)<m_size;
        const unsigned active_mask=(m_size==8)?0xffffffffu:((1u<<(m_size*4))-1u);
        unsigned b0=0,b1=0;
        if(active){
            const int act_block=int(group)*blocks_per_row+tile*2;
            b0=nqa4_decode_block_4lane(
                act_codes,act_beta_ids,act_betas,act_row_norms[group],act_block,
                inverse_sqrt_feature,active_mask);
            b1=nqa4_decode_block_4lane(
                act_codes,act_beta_ids,act_betas,act_row_norms[group],act_block+1,
                inverse_sqrt_feature,active_mask);
        }
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    const int token0=int(thread)*2;
    if(token0<m_size){
        output[token0*n_size+output0]=c0;
        output[token0*n_size+output1]=c2;
    }
    if(token0+1<m_size){
        output[(token0+1)*n_size+output0]=c1;
        output[(token0+1)*n_size+output1]=c3;
    }
}

torch::Tensor nqrt4_w4a4_mma_small_m_cuda(
    torch::Tensor act_codes,torch::Tensor act_beta_ids,
    torch::Tensor act_row_norms,torch::Tensor act_betas,
    torch::Tensor weight_codes,torch::Tensor weight_beta_ids,
    torch::Tensor weight_row_norms,torch::Tensor weight_betas,
    int64_t weight_offset,int64_t weight_seed) {
    check_cuda_contiguous(act_codes,"act_codes");
    check_cuda_contiguous(act_beta_ids,"act_beta_ids");
    check_cuda_contiguous(act_row_norms,"act_row_norms");
    check_cuda_contiguous(act_betas,"act_betas");
    check_cuda_contiguous(weight_codes,"weight_codes");
    check_cuda_contiguous(weight_beta_ids,"weight_beta_ids");
    check_cuda_contiguous(weight_row_norms,"weight_row_norms");
    check_cuda_contiguous(weight_betas,"weight_betas");
    check_same_device(act_codes,act_beta_ids,"act_beta_ids");
    check_same_device(act_codes,act_row_norms,"act_row_norms");
    check_same_device(act_codes,act_betas,"act_betas");
    check_same_device(act_codes,weight_codes,"weight_codes");
    check_same_device(act_codes,weight_beta_ids,"weight_beta_ids");
    check_same_device(act_codes,weight_row_norms,"weight_row_norms");
    check_same_device(act_codes,weight_betas,"weight_betas");
    TORCH_CHECK(act_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(act_betas.scalar_type()==torch::kFloat32&&act_betas.numel()==4);
    TORCH_CHECK(weight_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(weight_betas.scalar_type()==torch::kFloat32&&weight_betas.numel()==4);
    c10::cuda::CUDAGuard device_guard(act_codes.device());
    const int64_t m_size=act_row_norms.numel();
    TORCH_CHECK(m_size==2||m_size==4||m_size==8,"M must be 2, 4, or 8");
    const int64_t n_size=weight_row_norms.numel();
    TORCH_CHECK(n_size>0&&n_size%16==0,"N must be a positive multiple of 16");
    TORCH_CHECK(weight_codes.numel()%n_size==0,"weight payload must divide by N");
    const int64_t k_size=weight_codes.numel()*2/n_size;
    TORCH_CHECK(k_size>0&&k_size%16==0&&k_size<=INT_MAX,"K must be a supported multiple of 16");
    TORCH_CHECK(act_codes.numel()==m_size*k_size/2,"A4 code payload shape mismatch");
    TORCH_CHECK(act_beta_ids.numel()==m_size*k_size/32,"A4 beta payload shape mismatch");
    TORCH_CHECK(weight_beta_ids.numel()==n_size*k_size/32,"W4 beta payload shape mismatch");
    TORCH_CHECK(n_size<=INT_MAX);
    auto output=torch::empty({m_size,n_size},weight_row_norms.options());
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    nqrt4_w4a4_mma_small_m_kernel<<<unsigned(n_size/16),32,0,stream>>>(
        act_codes.data_ptr<uint8_t>(),act_beta_ids.data_ptr<uint8_t>(),
        act_row_norms.data_ptr<float>(),act_betas.data_ptr<float>(),
        weight_codes.data_ptr<uint8_t>(),weight_beta_ids.data_ptr<uint8_t>(),
        weight_row_norms.data_ptr<float>(),weight_betas.data_ptr<float>(),
        int(m_size),int(k_size),int(n_size),weight_offset,weight_seed,rsqrtf(float(k_size)),
        output.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 small-M transposed launch failed: ",cudaGetErrorString(error));
    return output;
}

// Correctness-first split-K probe for the M2 down projection. Each warp owns
// one [16 output channels, K/split_k] partition and writes deterministic FP32
// partial sums. A second kernel reduces partitions in increasing split order.
__global__ void nqrt4_w4a4_mma_small_m_splitk_partial_kernel(
    const uint8_t* act_codes,const uint8_t* act_beta_ids,
    const float* act_row_norms,const float* act_betas,
    const uint8_t* weight_codes,const uint8_t* weight_beta_ids,
    const float* weight_row_norms,const float* weight_betas,
    int m_size,int k_size,int n_size,int split_k,
    int64_t weight_offset,int64_t weight_seed,
    float inverse_sqrt_feature,float* partials) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3;
    const int output_base=int(blockIdx.x)*16;
    const int output0=output_base+int(group),output1=output0+8;
    const int split=int(blockIdx.y);
    const int total_tiles=k_size/16;
    const int tiles_per_split=total_tiles/split_k;
    const int tile_begin=split*tiles_per_split;
    const int tile_end=tile_begin+tiles_per_split;
    const int blocks_per_row=k_size/8;
    float c0=0,c1=0,c2=0,c3=0;
    const unsigned active_mask=(1u<<(m_size*4))-1u;
    for(int tile=tile_begin;tile<tile_end;tile++){
        const int weight_block0=output0*blocks_per_row+tile*2;
        const int weight_block1=output1*blocks_per_row+tile*2;
        const unsigned a0=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output0],weight_block0,
            weight_offset+weight_block0,weight_seed,inverse_sqrt_feature);
        const unsigned a1=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output1],weight_block1,
            weight_offset+weight_block1,weight_seed,inverse_sqrt_feature);
        const unsigned a2=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output0],weight_block0+1,
            weight_offset+weight_block0+1,weight_seed,inverse_sqrt_feature);
        const unsigned a3=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output1],weight_block1+1,
            weight_offset+weight_block1+1,weight_seed,inverse_sqrt_feature);
        unsigned b0=0,b1=0;
        if(int(group)<m_size){
            const int act_block=int(group)*blocks_per_row+tile*2;
            b0=nqa4_decode_block_4lane(
                act_codes,act_beta_ids,act_betas,act_row_norms[group],act_block,
                inverse_sqrt_feature,active_mask);
            b1=nqa4_decode_block_4lane(
                act_codes,act_beta_ids,act_betas,act_row_norms[group],act_block+1,
                inverse_sqrt_feature,active_mask);
        }
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    const int token0=int(thread)*2;
    const int64_t split_offset=int64_t(split)*m_size*n_size;
    if(token0<m_size){
        partials[split_offset+int64_t(token0)*n_size+output0]=c0;
        partials[split_offset+int64_t(token0)*n_size+output1]=c2;
    }
    if(token0+1<m_size){
        partials[split_offset+int64_t(token0+1)*n_size+output0]=c1;
        partials[split_offset+int64_t(token0+1)*n_size+output1]=c3;
    }
}

__global__ void nqrt4_w4a4_splitk_reduce_kernel(
    const float* partials,int elements,int split_k,float* output) {
    const int index=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    if(index>=elements)return;
    float value=0.0f;
    for(int split=0;split<split_k;split++){
        value+=partials[int64_t(split)*elements+index];
    }
    output[index]=value;
}

torch::Tensor nqrt4_w4a4_mma_small_m_splitk_probe_cuda(
    torch::Tensor act_codes,torch::Tensor act_beta_ids,
    torch::Tensor act_row_norms,torch::Tensor act_betas,
    torch::Tensor weight_codes,torch::Tensor weight_beta_ids,
    torch::Tensor weight_row_norms,torch::Tensor weight_betas,
    int64_t weight_offset,int64_t weight_seed,int64_t split_k) {
    check_cuda_contiguous(act_codes,"act_codes");
    check_cuda_contiguous(act_beta_ids,"act_beta_ids");
    check_cuda_contiguous(act_row_norms,"act_row_norms");
    check_cuda_contiguous(act_betas,"act_betas");
    check_cuda_contiguous(weight_codes,"weight_codes");
    check_cuda_contiguous(weight_beta_ids,"weight_beta_ids");
    check_cuda_contiguous(weight_row_norms,"weight_row_norms");
    check_cuda_contiguous(weight_betas,"weight_betas");
    check_same_device(act_codes,act_beta_ids,"act_beta_ids");
    check_same_device(act_codes,act_row_norms,"act_row_norms");
    check_same_device(act_codes,act_betas,"act_betas");
    check_same_device(act_codes,weight_codes,"weight_codes");
    check_same_device(act_codes,weight_beta_ids,"weight_beta_ids");
    check_same_device(act_codes,weight_row_norms,"weight_row_norms");
    check_same_device(act_codes,weight_betas,"weight_betas");
    TORCH_CHECK(act_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_row_norms.scalar_type()==torch::kFloat32&&act_row_norms.numel()==2,
        "split-K probe requires M=2");
    TORCH_CHECK(act_betas.scalar_type()==torch::kFloat32&&act_betas.numel()==4);
    TORCH_CHECK(weight_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(weight_betas.scalar_type()==torch::kFloat32&&weight_betas.numel()==4);
    TORCH_CHECK(split_k==2||split_k==4||split_k==8,
        "split_k must be 2, 4, or 8");
    c10::cuda::CUDAGuard device_guard(act_codes.device());
    const int64_t m_size=2;
    const int64_t n_size=weight_row_norms.numel();
    TORCH_CHECK(n_size>0&&n_size%16==0,"N must be a positive multiple of 16");
    TORCH_CHECK(weight_codes.numel()%n_size==0,"weight payload must divide by N");
    const int64_t k_size=weight_codes.numel()*2/n_size;
    TORCH_CHECK(k_size>0&&k_size%16==0&&k_size<=INT_MAX,
        "K must be a supported multiple of 16");
    TORCH_CHECK((k_size/16)%split_k==0,
        "K tiles must divide evenly by split_k");
    TORCH_CHECK(act_codes.numel()==m_size*k_size/2,"A4 code payload shape mismatch");
    TORCH_CHECK(act_beta_ids.numel()==m_size*k_size/32,"A4 beta payload shape mismatch");
    TORCH_CHECK(weight_beta_ids.numel()==n_size*k_size/32,"W4 beta payload shape mismatch");
    auto partials=torch::empty({split_k,m_size,n_size},weight_row_norms.options());
    auto output=torch::empty({m_size,n_size},weight_row_norms.options());
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    const dim3 grid(unsigned(n_size/16),unsigned(split_k),1);
    nqrt4_w4a4_mma_small_m_splitk_partial_kernel<<<grid,32,0,stream>>>(
        act_codes.data_ptr<uint8_t>(),act_beta_ids.data_ptr<uint8_t>(),
        act_row_norms.data_ptr<float>(),act_betas.data_ptr<float>(),
        weight_codes.data_ptr<uint8_t>(),weight_beta_ids.data_ptr<uint8_t>(),
        weight_row_norms.data_ptr<float>(),weight_betas.data_ptr<float>(),
        int(m_size),int(k_size),int(n_size),int(split_k),weight_offset,weight_seed,
        rsqrtf(float(k_size)),partials.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 split-K partial launch failed: ",cudaGetErrorString(error));
    const int elements=int(m_size*n_size);
    nqrt4_w4a4_splitk_reduce_kernel<<<unsigned((elements+255)/256),256,0,stream>>>(
        partials.data_ptr<float>(),elements,int(split_k),output.data_ptr<float>());
    error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 split-K reduce launch failed: ",cudaGetErrorString(error));
    return output;
}

// M32/M64/M128 structure probe. Four, eight, or sixteen warps share one CTA;
// each warp computes an independent 8-token slice against the same 16 output
// channels.
// This intentionally keeps weight reads warp-local: the first experiment
// isolates the multi-warp mapping before adding shared-memory staging.
__global__ void nqrt4_w4a4_mma_large_m_multiwarp_probe_kernel(
    const uint8_t* act_codes,const uint8_t* act_beta_ids,
    const float* act_row_norms,const float* act_betas,
    const uint8_t* weight_codes,const uint8_t* weight_beta_ids,
    const float* weight_row_norms,const float* weight_betas,
    int m_size,int k_size,int n_size,int64_t weight_offset,int64_t weight_seed,
    float inverse_sqrt_feature,float* output) {
    const unsigned warp=threadIdx.x>>5;
    const unsigned lane=threadIdx.x&31;
    const unsigned group=lane>>2,thread=lane&3;
    const int token_base=int(warp)*8;
    const int output_base=int(blockIdx.x)*16;
    const int output0=output_base+int(group),output1=output0+8;
    const int blocks_per_row=k_size/8;
    float c0=0,c1=0,c2=0,c3=0;
    for(int tile=0;tile<k_size/16;tile++){
        const int weight_block0=output0*blocks_per_row+tile*2;
        const int weight_block1=output1*blocks_per_row+tile*2;
        const unsigned a0=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output0],weight_block0,
            weight_offset+weight_block0,weight_seed,inverse_sqrt_feature);
        const unsigned a1=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output1],weight_block1,
            weight_offset+weight_block1,weight_seed,inverse_sqrt_feature);
        const unsigned a2=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output0],weight_block0+1,
            weight_offset+weight_block0+1,weight_seed,inverse_sqrt_feature);
        const unsigned a3=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output1],weight_block1+1,
            weight_offset+weight_block1+1,weight_seed,inverse_sqrt_feature);
        const int token=token_base+int(group);
        const int act_block=token*blocks_per_row+tile*2;
        const unsigned b0=nqa4_decode_block_4lane(
            act_codes,act_beta_ids,act_betas,act_row_norms[token],act_block,
            inverse_sqrt_feature);
        const unsigned b1=nqa4_decode_block_4lane(
            act_codes,act_beta_ids,act_betas,act_row_norms[token],act_block+1,
            inverse_sqrt_feature);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    const int token0=token_base+int(thread)*2;
    output[token0*n_size+output0]=c0;
    output[token0*n_size+output1]=c2;
    output[(token0+1)*n_size+output0]=c1;
    output[(token0+1)*n_size+output1]=c3;
}

torch::Tensor nqrt4_w4a4_mma_m32_multiwarp_probe_cuda(
    torch::Tensor act_codes,torch::Tensor act_beta_ids,
    torch::Tensor act_row_norms,torch::Tensor act_betas,
    torch::Tensor weight_codes,torch::Tensor weight_beta_ids,
    torch::Tensor weight_row_norms,torch::Tensor weight_betas,
    int64_t weight_offset,int64_t weight_seed) {
    check_cuda_contiguous(act_codes,"act_codes");
    check_cuda_contiguous(act_beta_ids,"act_beta_ids");
    check_cuda_contiguous(act_row_norms,"act_row_norms");
    check_cuda_contiguous(act_betas,"act_betas");
    check_cuda_contiguous(weight_codes,"weight_codes");
    check_cuda_contiguous(weight_beta_ids,"weight_beta_ids");
    check_cuda_contiguous(weight_row_norms,"weight_row_norms");
    check_cuda_contiguous(weight_betas,"weight_betas");
    check_same_device(act_codes,act_beta_ids,"act_beta_ids");
    check_same_device(act_codes,act_row_norms,"act_row_norms");
    check_same_device(act_codes,act_betas,"act_betas");
    check_same_device(act_codes,weight_codes,"weight_codes");
    check_same_device(act_codes,weight_beta_ids,"weight_beta_ids");
    check_same_device(act_codes,weight_row_norms,"weight_row_norms");
    check_same_device(act_codes,weight_betas,"weight_betas");
    TORCH_CHECK(act_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(act_betas.scalar_type()==torch::kFloat32&&act_betas.numel()==4);
    TORCH_CHECK(weight_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(weight_betas.scalar_type()==torch::kFloat32&&weight_betas.numel()==4);
    c10::cuda::CUDAGuard device_guard(act_codes.device());
    const int64_t m_size=act_row_norms.numel();
    TORCH_CHECK(m_size==32||m_size==64||m_size==128,"M must be 32, 64, or 128");
    const int64_t n_size=weight_row_norms.numel();
    TORCH_CHECK(n_size>0&&n_size%16==0,"N must be a positive multiple of 16");
    TORCH_CHECK(weight_codes.numel()%n_size==0,"weight payload must divide by N");
    const int64_t k_size=weight_codes.numel()*2/n_size;
    TORCH_CHECK(k_size>0&&k_size%16==0&&k_size<=INT_MAX,"K must be a supported multiple of 16");
    TORCH_CHECK(act_codes.numel()==m_size*k_size/2,"A4 code payload shape mismatch");
    TORCH_CHECK(act_beta_ids.numel()==m_size*k_size/32,"A4 beta payload shape mismatch");
    TORCH_CHECK(weight_beta_ids.numel()==n_size*k_size/32,"W4 beta payload shape mismatch");
    TORCH_CHECK(n_size<=INT_MAX);
    auto output=torch::empty({m_size,n_size},weight_row_norms.options());
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    nqrt4_w4a4_mma_large_m_multiwarp_probe_kernel<<<
        unsigned(n_size/16),unsigned(m_size*4),0,stream>>>(
        act_codes.data_ptr<uint8_t>(),act_beta_ids.data_ptr<uint8_t>(),
        act_row_norms.data_ptr<float>(),act_betas.data_ptr<float>(),
        weight_codes.data_ptr<uint8_t>(),weight_beta_ids.data_ptr<uint8_t>(),
        weight_row_norms.data_ptr<float>(),weight_betas.data_ptr<float>(),
        int(m_size),int(k_size),int(n_size),weight_offset,weight_seed,rsqrtf(float(k_size)),
        output.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 large-M multi-warp launch failed: ",cudaGetErrorString(error));
    return output;
}

// Correctness-first M128 decoded-weight reuse probe. Warp 0 produces one
// 16x16 BF16 weight fragment per K tile; all sixteen warps consume it from
// shared memory. Two CTA barriers intentionally make lifetime rules explicit.
__global__ void nqrt4_w4a4_mma_m128_staged_probe_kernel(
    const uint8_t* act_codes,const uint8_t* act_beta_ids,
    const float* act_row_norms,const float* act_betas,
    const uint8_t* weight_codes,const uint8_t* weight_beta_ids,
    const float* weight_row_norms,const float* weight_betas,
    int k_size,int n_size,int64_t weight_offset,int64_t weight_seed,
    float inverse_sqrt_feature,float* output) {
    __shared__ unsigned weight_fragment[4][32];
    const unsigned warp=threadIdx.x>>5;
    const unsigned lane=threadIdx.x&31;
    const unsigned group=lane>>2,thread=lane&3;
    const int token_base=int(warp)*8;
    const int output_base=int(blockIdx.x)*16;
    const int output0=output_base+int(group),output1=output0+8;
    const int blocks_per_row=k_size/8;
    float c0=0,c1=0,c2=0,c3=0;
    for(int tile=0;tile<k_size/16;tile++){
        if(warp==0){
            const int weight_block0=output0*blocks_per_row+tile*2;
            const int weight_block1=output1*blocks_per_row+tile*2;
            weight_fragment[0][lane]=nqrt4_decode_block_4lane(
                weight_codes,weight_beta_ids,weight_betas,
                weight_row_norms[output0],weight_block0,
                weight_offset+weight_block0,weight_seed,inverse_sqrt_feature);
            weight_fragment[1][lane]=nqrt4_decode_block_4lane(
                weight_codes,weight_beta_ids,weight_betas,
                weight_row_norms[output1],weight_block1,
                weight_offset+weight_block1,weight_seed,inverse_sqrt_feature);
            weight_fragment[2][lane]=nqrt4_decode_block_4lane(
                weight_codes,weight_beta_ids,weight_betas,
                weight_row_norms[output0],weight_block0+1,
                weight_offset+weight_block0+1,weight_seed,inverse_sqrt_feature);
            weight_fragment[3][lane]=nqrt4_decode_block_4lane(
                weight_codes,weight_beta_ids,weight_betas,
                weight_row_norms[output1],weight_block1+1,
                weight_offset+weight_block1+1,weight_seed,inverse_sqrt_feature);
        }
        __syncthreads();
        const unsigned a0=weight_fragment[0][lane];
        const unsigned a1=weight_fragment[1][lane];
        const unsigned a2=weight_fragment[2][lane];
        const unsigned a3=weight_fragment[3][lane];
        const int token=token_base+int(group);
        const int act_block=token*blocks_per_row+tile*2;
        const unsigned b0=nqa4_decode_block_4lane(
            act_codes,act_beta_ids,act_betas,act_row_norms[token],act_block,
            inverse_sqrt_feature);
        const unsigned b1=nqa4_decode_block_4lane(
            act_codes,act_beta_ids,act_betas,act_row_norms[token],act_block+1,
            inverse_sqrt_feature);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        __syncthreads();
    }
    const int token0=token_base+int(thread)*2;
    output[token0*n_size+output0]=c0;
    output[token0*n_size+output1]=c2;
    output[(token0+1)*n_size+output0]=c1;
    output[(token0+1)*n_size+output1]=c3;
}

torch::Tensor nqrt4_w4a4_mma_m128_staged_probe_cuda(
    torch::Tensor act_codes,torch::Tensor act_beta_ids,
    torch::Tensor act_row_norms,torch::Tensor act_betas,
    torch::Tensor weight_codes,torch::Tensor weight_beta_ids,
    torch::Tensor weight_row_norms,torch::Tensor weight_betas,
    int64_t weight_offset,int64_t weight_seed) {
    check_cuda_contiguous(act_codes,"act_codes");
    check_cuda_contiguous(act_beta_ids,"act_beta_ids");
    check_cuda_contiguous(act_row_norms,"act_row_norms");
    check_cuda_contiguous(act_betas,"act_betas");
    check_cuda_contiguous(weight_codes,"weight_codes");
    check_cuda_contiguous(weight_beta_ids,"weight_beta_ids");
    check_cuda_contiguous(weight_row_norms,"weight_row_norms");
    check_cuda_contiguous(weight_betas,"weight_betas");
    check_same_device(act_codes,act_beta_ids,"act_beta_ids");
    check_same_device(act_codes,act_row_norms,"act_row_norms");
    check_same_device(act_codes,act_betas,"act_betas");
    check_same_device(act_codes,weight_codes,"weight_codes");
    check_same_device(act_codes,weight_beta_ids,"weight_beta_ids");
    check_same_device(act_codes,weight_row_norms,"weight_row_norms");
    check_same_device(act_codes,weight_betas,"weight_betas");
    TORCH_CHECK(act_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(act_betas.scalar_type()==torch::kFloat32&&act_betas.numel()==4);
    TORCH_CHECK(weight_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(weight_betas.scalar_type()==torch::kFloat32&&weight_betas.numel()==4);
    c10::cuda::CUDAGuard device_guard(act_codes.device());
    const int64_t m_size=act_row_norms.numel();
    TORCH_CHECK(m_size==32||m_size==64||m_size==128,"M must be 32, 64, or 128");
    const int64_t n_size=weight_row_norms.numel();
    TORCH_CHECK(n_size>0&&n_size%16==0,"N must be a positive multiple of 16");
    TORCH_CHECK(weight_codes.numel()%n_size==0,"weight payload must divide by N");
    const int64_t k_size=weight_codes.numel()*2/n_size;
    TORCH_CHECK(k_size>0&&k_size%16==0&&k_size<=INT_MAX,"K must be a supported multiple of 16");
    TORCH_CHECK(act_codes.numel()==m_size*k_size/2,"A4 code payload shape mismatch");
    TORCH_CHECK(act_beta_ids.numel()==m_size*k_size/32,"A4 beta payload shape mismatch");
    TORCH_CHECK(weight_beta_ids.numel()==n_size*k_size/32,"W4 beta payload shape mismatch");
    TORCH_CHECK(n_size<=INT_MAX);
    auto output=torch::empty({m_size,n_size},weight_row_norms.options());
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    nqrt4_w4a4_mma_m128_staged_probe_kernel<<<
        unsigned(n_size/16),unsigned(m_size*4),0,stream>>>(
        act_codes.data_ptr<uint8_t>(),act_beta_ids.data_ptr<uint8_t>(),
        act_row_norms.data_ptr<float>(),act_betas.data_ptr<float>(),
        weight_codes.data_ptr<uint8_t>(),weight_beta_ids.data_ptr<uint8_t>(),
        weight_row_norms.data_ptr<float>(),weight_betas.data_ptr<float>(),
        int(k_size),int(n_size),weight_offset,weight_seed,rsqrtf(float(k_size)),
        output.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 large-M staged launch failed: ",cudaGetErrorString(error));
    return output;
}

// Double-buffered warp-specialized M128 probe. Warp 0 produces the next
// decoded weight tile while warps 1-16 consume the current tile. Alternating
// shared-memory buffers reduce the steady-state protocol to one CTA barrier
// per K tile and overlap weight decode with activation decode plus MMA.
__global__ void nqrt4_w4a4_mma_m128_pipeline_probe_kernel(
    const uint8_t* act_codes,const uint8_t* act_beta_ids,
    const float* act_row_norms,const float* act_betas,
    const uint8_t* weight_codes,const uint8_t* weight_beta_ids,
    const float* weight_row_norms,const float* weight_betas,
    int k_size,int n_size,int64_t weight_offset,int64_t weight_seed,
    float inverse_sqrt_feature,float* output) {
    __shared__ unsigned weight_fragment[2][4][32];
    const unsigned warp=threadIdx.x>>5;
    const unsigned lane=threadIdx.x&31;
    const unsigned group=lane>>2,thread=lane&3;
    const bool producer=warp==0;
    const int consumer_warp=int(warp)-1;
    const int output_base=int(blockIdx.x)*16;
    const int output0=output_base+int(group),output1=output0+8;
    const int blocks_per_row=k_size/8;
    const int tile_count=k_size/16;
    auto produce_tile=[&](int tile,int stage){
        const int weight_block0=output0*blocks_per_row+tile*2;
        const int weight_block1=output1*blocks_per_row+tile*2;
        weight_fragment[stage][0][lane]=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output0],weight_block0,
            weight_offset+weight_block0,weight_seed,inverse_sqrt_feature);
        weight_fragment[stage][1][lane]=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output1],weight_block1,
            weight_offset+weight_block1,weight_seed,inverse_sqrt_feature);
        weight_fragment[stage][2][lane]=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output0],weight_block0+1,
            weight_offset+weight_block0+1,weight_seed,inverse_sqrt_feature);
        weight_fragment[stage][3][lane]=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output1],weight_block1+1,
            weight_offset+weight_block1+1,weight_seed,inverse_sqrt_feature);
    };
    if(producer)produce_tile(0,0);
    __syncthreads();
    float c0=0,c1=0,c2=0,c3=0;
    for(int tile=0;tile<tile_count;tile++){
        const int stage=tile&1;
        if(producer){
            if(tile+1<tile_count)produce_tile(tile+1,stage^1);
        }else{
            const unsigned a0=weight_fragment[stage][0][lane];
            const unsigned a1=weight_fragment[stage][1][lane];
            const unsigned a2=weight_fragment[stage][2][lane];
            const unsigned a3=weight_fragment[stage][3][lane];
            const int token=consumer_warp*8+int(group);
            const int act_block=token*blocks_per_row+tile*2;
            const unsigned b0=nqa4_decode_block_4lane(
                act_codes,act_beta_ids,act_betas,act_row_norms[token],act_block,
                inverse_sqrt_feature);
            const unsigned b1=nqa4_decode_block_4lane(
                act_codes,act_beta_ids,act_betas,act_row_norms[token],act_block+1,
                inverse_sqrt_feature);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
        __syncthreads();
    }
    if(!producer){
        const int token0=consumer_warp*8+int(thread)*2;
        output[token0*n_size+output0]=c0;
        output[token0*n_size+output1]=c2;
        output[(token0+1)*n_size+output0]=c1;
        output[(token0+1)*n_size+output1]=c3;
    }
}

torch::Tensor nqrt4_w4a4_mma_m128_pipeline_probe_cuda(
    torch::Tensor act_codes,torch::Tensor act_beta_ids,
    torch::Tensor act_row_norms,torch::Tensor act_betas,
    torch::Tensor weight_codes,torch::Tensor weight_beta_ids,
    torch::Tensor weight_row_norms,torch::Tensor weight_betas,
    int64_t weight_offset,int64_t weight_seed) {
    check_cuda_contiguous(act_codes,"act_codes");
    check_cuda_contiguous(act_beta_ids,"act_beta_ids");
    check_cuda_contiguous(act_row_norms,"act_row_norms");
    check_cuda_contiguous(act_betas,"act_betas");
    check_cuda_contiguous(weight_codes,"weight_codes");
    check_cuda_contiguous(weight_beta_ids,"weight_beta_ids");
    check_cuda_contiguous(weight_row_norms,"weight_row_norms");
    check_cuda_contiguous(weight_betas,"weight_betas");
    check_same_device(act_codes,act_beta_ids,"act_beta_ids");
    check_same_device(act_codes,act_row_norms,"act_row_norms");
    check_same_device(act_codes,act_betas,"act_betas");
    check_same_device(act_codes,weight_codes,"weight_codes");
    check_same_device(act_codes,weight_beta_ids,"weight_beta_ids");
    check_same_device(act_codes,weight_row_norms,"weight_row_norms");
    check_same_device(act_codes,weight_betas,"weight_betas");
    TORCH_CHECK(act_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_row_norms.scalar_type()==torch::kFloat32&&act_row_norms.numel()==128);
    TORCH_CHECK(act_betas.scalar_type()==torch::kFloat32&&act_betas.numel()==4);
    TORCH_CHECK(weight_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(weight_betas.scalar_type()==torch::kFloat32&&weight_betas.numel()==4);
    c10::cuda::CUDAGuard device_guard(act_codes.device());
    const int64_t n_size=weight_row_norms.numel();
    TORCH_CHECK(n_size>0&&n_size%16==0,"N must be a positive multiple of 16");
    TORCH_CHECK(weight_codes.numel()%n_size==0,"weight payload must divide by N");
    const int64_t k_size=weight_codes.numel()*2/n_size;
    TORCH_CHECK(k_size>0&&k_size%16==0&&k_size<=INT_MAX,"K must be a supported multiple of 16");
    TORCH_CHECK(act_codes.numel()==128*k_size/2,"A4 code payload shape mismatch");
    TORCH_CHECK(act_beta_ids.numel()==128*k_size/32,"A4 beta payload shape mismatch");
    TORCH_CHECK(weight_beta_ids.numel()==n_size*k_size/32,"W4 beta payload shape mismatch");
    TORCH_CHECK(n_size<=INT_MAX);
    auto output=torch::empty({128,n_size},weight_row_norms.options());
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    nqrt4_w4a4_mma_m128_pipeline_probe_kernel<<<unsigned(n_size/16),544,0,stream>>>(
        act_codes.data_ptr<uint8_t>(),act_beta_ids.data_ptr<uint8_t>(),
        act_row_norms.data_ptr<float>(),act_betas.data_ptr<float>(),
        weight_codes.data_ptr<uint8_t>(),weight_beta_ids.data_ptr<uint8_t>(),
        weight_row_norms.data_ptr<float>(),weight_betas.data_ptr<float>(),
        int(k_size),int(n_size),weight_offset,weight_seed,rsqrtf(float(k_size)),
        output.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 M128 pipeline launch failed: ",cudaGetErrorString(error));
    return output;
}

__device__ __forceinline__ float nqrt4_unpack_bf16(unsigned pair,int high) {
    const unsigned bits=high?(pair>>16):(pair&0xffffu);
    return __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(bits)));
}

// Correctness-first direct-packed M1 GEMV. One four-lane subgroup owns one
// output channel; every lane decodes and accumulates two coordinates per E8
// block, then the subgroup reduces its four partial sums.
__global__ void nqrt4_w4a4_gemv_m1_kernel(
    const uint8_t* act_codes,const uint8_t* act_beta_ids,
    const float* act_row_norms,const float* act_betas,
    const uint8_t* weight_codes,const uint8_t* weight_beta_ids,
    const float* weight_row_norms,const float* weight_betas,
    int k_size,int n_size,int64_t weight_offset,int64_t weight_seed,
    float inverse_sqrt_feature,float* output) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3;
    const int output_channel=int(blockIdx.x)*8+int(group);
    const int blocks_per_row=k_size/8;
    float accumulator=0.0f;
    #pragma unroll 1
    for(int block=0;block<blocks_per_row;block++){
        const unsigned activation=nqa4_decode_block_4lane(
            act_codes,act_beta_ids,act_betas,act_row_norms[0],block,
            inverse_sqrt_feature);
        const int weight_block=output_channel*blocks_per_row+block;
        const unsigned weight=nqrt4_decode_block_4lane(
            weight_codes,weight_beta_ids,weight_betas,
            weight_row_norms[output_channel],weight_block,
            weight_offset+weight_block,weight_seed,inverse_sqrt_feature);
        accumulator=fmaf(
            nqrt4_unpack_bf16(activation,0),nqrt4_unpack_bf16(weight,0),
            accumulator);
        accumulator=fmaf(
            nqrt4_unpack_bf16(activation,1),nqrt4_unpack_bf16(weight,1),
            accumulator);
    }
    accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,2,4);
    accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,1,4);
    if(thread==0)output[output_channel]=accumulator;
}

// Experimental packed-word decoder probe.  The activation codeword/beta byte
// is identical for all eight output-channel groups, so lane 0 loads it once
// for the entire warp.  Each weight codeword/beta byte is loaded once per
// four-lane subgroup.  Keeping this as a separate symbol makes profiler and
// SASS comparisons possible without changing the production M1 kernel.
__global__ void nqrt4_w4a4_gemv_m1_packed_word_probe_kernel(
    const uint8_t* act_codes,const uint8_t* act_beta_ids,
    const float* act_row_norms,const float* act_betas,
    const uint8_t* weight_codes,const uint8_t* weight_beta_ids,
    const float* weight_row_norms,const float* weight_betas,
    int k_size,int n_size,int64_t weight_offset,int64_t weight_seed,
    float inverse_sqrt_feature,float* output) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3;
    const int output_channel=int(blockIdx.x)*8+int(group);
    const int blocks_per_row=k_size/8;
    float accumulator=0.0f;
    #pragma unroll 1
    for(int block=0;block<blocks_per_row;block++){
        unsigned activation_word=0,activation_beta_pack=0;
        if(lane==0){
            activation_word=reinterpret_cast<const unsigned*>(act_codes)[block];
            activation_beta_pack=act_beta_ids[block>>2];
        }
        activation_word=__shfl_sync(0xffffffffu,activation_word,0);
        activation_beta_pack=__shfl_sync(0xffffffffu,activation_beta_pack,0);
        const unsigned activation=nqa4_decode_codeword_4lane(
            activation_word,activation_beta_pack,act_betas,act_row_norms[0],
            block,inverse_sqrt_feature);

        const int weight_block=output_channel*blocks_per_row+block;
        unsigned weight_word=0,weight_beta_pack=0;
        if(thread==0){
            weight_word=reinterpret_cast<const unsigned*>(weight_codes)[weight_block];
            weight_beta_pack=weight_beta_ids[weight_block>>2];
        }
        weight_word=__shfl_sync(0xffffffffu,weight_word,0,4);
        weight_beta_pack=__shfl_sync(0xffffffffu,weight_beta_pack,0,4);
        const unsigned weight=nqrt4_decode_codeword_4lane(
            weight_word,weight_beta_pack,weight_betas,
            weight_row_norms[output_channel],weight_block,
            weight_offset+weight_block,weight_seed,inverse_sqrt_feature);
        accumulator=fmaf(
            nqrt4_unpack_bf16(activation,0),nqrt4_unpack_bf16(weight,0),
            accumulator);
        accumulator=fmaf(
            nqrt4_unpack_bf16(activation,1),nqrt4_unpack_bf16(weight,1),
            accumulator);
    }
    accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,2,4);
    accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,1,4);
    if(thread==0)output[output_channel]=accumulator;
}

// Long-K M1 probe. Each grid-y partition accumulates a deterministic K slice
// for eight output channels; the existing fixed-order reducer combines FP32
// partials after every partition completes.
__global__ void nqrt4_w4a4_gemv_m1_splitk_partial_kernel(
    const uint8_t* act_codes,const uint8_t* act_beta_ids,
    const float* act_row_norms,const float* act_betas,
    const uint8_t* weight_codes,const uint8_t* weight_beta_ids,
    const float* weight_row_norms,const float* weight_betas,
    int k_size,int n_size,int split_k,int64_t weight_offset,int64_t weight_seed,
    float inverse_sqrt_feature,float* partials) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3;
    const int output_channel=int(blockIdx.x)*8+int(group);
    const int split=int(blockIdx.y);
    const int blocks_per_row=k_size/8;
    const int blocks_per_split=blocks_per_row/split_k;
    const int first_block=split*blocks_per_split;
    const int last_block=first_block+blocks_per_split;
    float accumulator=0.0f;
    #pragma unroll 1
    for(int block=first_block;block<last_block;block++){
        unsigned activation_word=0,activation_beta_pack=0;
        if(lane==0){
            activation_word=reinterpret_cast<const unsigned*>(act_codes)[block];
            activation_beta_pack=act_beta_ids[block>>2];
        }
        activation_word=__shfl_sync(0xffffffffu,activation_word,0);
        activation_beta_pack=__shfl_sync(0xffffffffu,activation_beta_pack,0);
        const unsigned activation=nqa4_decode_codeword_4lane(
            activation_word,activation_beta_pack,act_betas,act_row_norms[0],
            block,inverse_sqrt_feature);
        const int weight_block=output_channel*blocks_per_row+block;
        unsigned weight_word=0,weight_beta_pack=0;
        if(thread==0){
            weight_word=reinterpret_cast<const unsigned*>(weight_codes)[weight_block];
            weight_beta_pack=weight_beta_ids[weight_block>>2];
        }
        weight_word=__shfl_sync(0xffffffffu,weight_word,0,4);
        weight_beta_pack=__shfl_sync(0xffffffffu,weight_beta_pack,0,4);
        const unsigned weight=nqrt4_decode_codeword_4lane(
            weight_word,weight_beta_pack,weight_betas,
            weight_row_norms[output_channel],weight_block,
            weight_offset+weight_block,weight_seed,inverse_sqrt_feature);
        accumulator=fmaf(
            nqrt4_unpack_bf16(activation,0),nqrt4_unpack_bf16(weight,0),
            accumulator);
        accumulator=fmaf(
            nqrt4_unpack_bf16(activation,1),nqrt4_unpack_bf16(weight,1),
            accumulator);
    }
    accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,2,4);
    accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,1,4);
    if(thread==0)partials[split*n_size+output_channel]=accumulator;
}

// Diagnostic-only Kahan variant. This deliberately separates multiplication
// from addition so the compensation term can recover low-order addends lost
// inside each independent split. It is not used by production dispatch.
__global__ void nqrt4_w4a4_gemv_m1_splitk_compensated_partial_kernel(
    const uint8_t* act_codes,const uint8_t* act_beta_ids,
    const float* act_row_norms,const float* act_betas,
    const uint8_t* weight_codes,const uint8_t* weight_beta_ids,
    const float* weight_row_norms,const float* weight_betas,
    int k_size,int n_size,int split_k,int64_t weight_offset,int64_t weight_seed,
    float inverse_sqrt_feature,float* partials) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3;
    const int output_channel=int(blockIdx.x)*8+int(group);
    const int split=int(blockIdx.y);
    const int blocks_per_row=k_size/8;
    const int blocks_per_split=blocks_per_row/split_k;
    const int first_block=split*blocks_per_split;
    const int last_block=first_block+blocks_per_split;
    float accumulator=0.0f,compensation=0.0f;
    #pragma unroll 1
    for(int block=first_block;block<last_block;block++){
        unsigned activation_word=0,activation_beta_pack=0;
        if(lane==0){
            activation_word=reinterpret_cast<const unsigned*>(act_codes)[block];
            activation_beta_pack=act_beta_ids[block>>2];
        }
        activation_word=__shfl_sync(0xffffffffu,activation_word,0);
        activation_beta_pack=__shfl_sync(0xffffffffu,activation_beta_pack,0);
        const unsigned activation=nqa4_decode_codeword_4lane(
            activation_word,activation_beta_pack,act_betas,act_row_norms[0],
            block,inverse_sqrt_feature);
        const int weight_block=output_channel*blocks_per_row+block;
        unsigned weight_word=0,weight_beta_pack=0;
        if(thread==0){
            weight_word=reinterpret_cast<const unsigned*>(weight_codes)[weight_block];
            weight_beta_pack=weight_beta_ids[weight_block>>2];
        }
        weight_word=__shfl_sync(0xffffffffu,weight_word,0,4);
        weight_beta_pack=__shfl_sync(0xffffffffu,weight_beta_pack,0,4);
        const unsigned weight=nqrt4_decode_codeword_4lane(
            weight_word,weight_beta_pack,weight_betas,
            weight_row_norms[output_channel],weight_block,
            weight_offset+weight_block,weight_seed,inverse_sqrt_feature);
        const float product0=nqrt4_unpack_bf16(activation,0)
            *nqrt4_unpack_bf16(weight,0);
        const float adjusted0=product0-compensation;
        const float updated0=accumulator+adjusted0;
        compensation=(updated0-accumulator)-adjusted0;
        accumulator=updated0;
        const float product1=nqrt4_unpack_bf16(activation,1)
            *nqrt4_unpack_bf16(weight,1);
        const float adjusted1=product1-compensation;
        const float updated1=accumulator+adjusted1;
        compensation=(updated1-accumulator)-adjusted1;
        accumulator=updated1;
    }
    accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,2,4);
    accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,1,4);
    if(thread==0)partials[split*n_size+output_channel]=accumulator;
}

torch::Tensor nqrt4_w4a4_gemv_m1_legacy_cuda(
    torch::Tensor act_codes,torch::Tensor act_beta_ids,
    torch::Tensor act_row_norms,torch::Tensor act_betas,
    torch::Tensor weight_codes,torch::Tensor weight_beta_ids,
    torch::Tensor weight_row_norms,torch::Tensor weight_betas,
    int64_t weight_offset,int64_t weight_seed) {
    check_cuda_contiguous(act_codes,"act_codes");
    check_cuda_contiguous(act_beta_ids,"act_beta_ids");
    check_cuda_contiguous(act_row_norms,"act_row_norms");
    check_cuda_contiguous(act_betas,"act_betas");
    check_cuda_contiguous(weight_codes,"weight_codes");
    check_cuda_contiguous(weight_beta_ids,"weight_beta_ids");
    check_cuda_contiguous(weight_row_norms,"weight_row_norms");
    check_cuda_contiguous(weight_betas,"weight_betas");
    check_same_device(act_codes,act_beta_ids,"act_beta_ids");
    check_same_device(act_codes,act_row_norms,"act_row_norms");
    check_same_device(act_codes,act_betas,"act_betas");
    check_same_device(act_codes,weight_codes,"weight_codes");
    check_same_device(act_codes,weight_beta_ids,"weight_beta_ids");
    check_same_device(act_codes,weight_row_norms,"weight_row_norms");
    check_same_device(act_codes,weight_betas,"weight_betas");
    TORCH_CHECK(act_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_row_norms.scalar_type()==torch::kFloat32&&act_row_norms.numel()==1);
    TORCH_CHECK(act_betas.scalar_type()==torch::kFloat32&&act_betas.numel()==4);
    TORCH_CHECK(weight_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(weight_betas.scalar_type()==torch::kFloat32&&weight_betas.numel()==4);
    c10::cuda::CUDAGuard device_guard(act_codes.device());
    const int64_t n_size=weight_row_norms.numel();
    TORCH_CHECK(n_size>0&&n_size%8==0,"N must be a positive multiple of 8");
    TORCH_CHECK(weight_codes.numel()%n_size==0,"weight payload must divide by N");
    const int64_t k_size=weight_codes.numel()*2/n_size;
    TORCH_CHECK(k_size>0&&k_size%8==0&&k_size<=INT_MAX,"K must be a supported multiple of 8");
    TORCH_CHECK(act_codes.numel()==k_size/2,"A4 code payload shape mismatch");
    TORCH_CHECK(act_beta_ids.numel()==k_size/32,"A4 beta payload shape mismatch");
    TORCH_CHECK(weight_beta_ids.numel()==n_size*k_size/32,"W4 beta payload shape mismatch");
    TORCH_CHECK(n_size<=INT_MAX);
    auto output=torch::empty({1,n_size},weight_row_norms.options());
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    nqrt4_w4a4_gemv_m1_kernel<<<unsigned(n_size/8),32,0,stream>>>(
        act_codes.data_ptr<uint8_t>(),act_beta_ids.data_ptr<uint8_t>(),
        act_row_norms.data_ptr<float>(),act_betas.data_ptr<float>(),
        weight_codes.data_ptr<uint8_t>(),weight_beta_ids.data_ptr<uint8_t>(),
        weight_row_norms.data_ptr<float>(),weight_betas.data_ptr<float>(),
        int(k_size),int(n_size),weight_offset,weight_seed,rsqrtf(float(k_size)),
        output.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 M1 GEMV launch failed: ",cudaGetErrorString(error));
    return output;
}

torch::Tensor nqrt4_w4a4_gemv_m1_cuda(
    torch::Tensor act_codes,torch::Tensor act_beta_ids,
    torch::Tensor act_row_norms,torch::Tensor act_betas,
    torch::Tensor weight_codes,torch::Tensor weight_beta_ids,
    torch::Tensor weight_row_norms,torch::Tensor weight_betas,
    int64_t weight_offset,int64_t weight_seed) {
    check_cuda_contiguous(act_codes,"act_codes");
    check_cuda_contiguous(act_beta_ids,"act_beta_ids");
    check_cuda_contiguous(act_row_norms,"act_row_norms");
    check_cuda_contiguous(act_betas,"act_betas");
    check_cuda_contiguous(weight_codes,"weight_codes");
    check_cuda_contiguous(weight_beta_ids,"weight_beta_ids");
    check_cuda_contiguous(weight_row_norms,"weight_row_norms");
    check_cuda_contiguous(weight_betas,"weight_betas");
    check_same_device(act_codes,act_beta_ids,"act_beta_ids");
    check_same_device(act_codes,act_row_norms,"act_row_norms");
    check_same_device(act_codes,act_betas,"act_betas");
    check_same_device(act_codes,weight_codes,"weight_codes");
    check_same_device(act_codes,weight_beta_ids,"weight_beta_ids");
    check_same_device(act_codes,weight_row_norms,"weight_row_norms");
    check_same_device(act_codes,weight_betas,"weight_betas");
    TORCH_CHECK(act_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_row_norms.scalar_type()==torch::kFloat32&&act_row_norms.numel()==1);
    TORCH_CHECK(act_betas.scalar_type()==torch::kFloat32&&act_betas.numel()==4);
    TORCH_CHECK(weight_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(weight_betas.scalar_type()==torch::kFloat32&&weight_betas.numel()==4);
    c10::cuda::CUDAGuard device_guard(act_codes.device());
    const int64_t n_size=weight_row_norms.numel();
    TORCH_CHECK(n_size>0&&n_size%8==0,"N must be a positive multiple of 8");
    TORCH_CHECK(weight_codes.numel()%n_size==0,"weight payload must divide by N");
    const int64_t k_size=weight_codes.numel()*2/n_size;
    TORCH_CHECK(k_size>0&&k_size%8==0&&k_size<=INT_MAX,"K must be a supported multiple of 8");
    TORCH_CHECK(act_codes.numel()==k_size/2,"A4 code payload shape mismatch");
    TORCH_CHECK(act_beta_ids.numel()==k_size/32,"A4 beta payload shape mismatch");
    TORCH_CHECK(weight_beta_ids.numel()==n_size*k_size/32,"W4 beta payload shape mismatch");
    TORCH_CHECK(n_size<=INT_MAX);
    auto output=torch::empty({1,n_size},weight_row_norms.options());
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    nqrt4_w4a4_gemv_m1_packed_word_probe_kernel<<<unsigned(n_size/8),32,0,stream>>>(
        act_codes.data_ptr<uint8_t>(),act_beta_ids.data_ptr<uint8_t>(),
        act_row_norms.data_ptr<float>(),act_betas.data_ptr<float>(),
        weight_codes.data_ptr<uint8_t>(),weight_beta_ids.data_ptr<uint8_t>(),
        weight_row_norms.data_ptr<float>(),weight_betas.data_ptr<float>(),
        int(k_size),int(n_size),weight_offset,weight_seed,rsqrtf(float(k_size)),
        output.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 M1 packed-word launch failed: ",cudaGetErrorString(error));
    return output;
}

torch::Tensor nqrt4_w4a4_gemv_m1_splitk_probe_cuda(
    torch::Tensor act_codes,torch::Tensor act_beta_ids,
    torch::Tensor act_row_norms,torch::Tensor act_betas,
    torch::Tensor weight_codes,torch::Tensor weight_beta_ids,
    torch::Tensor weight_row_norms,torch::Tensor weight_betas,
    int64_t weight_offset,int64_t weight_seed,int64_t split_k) {
    check_cuda_contiguous(act_codes,"act_codes");
    check_cuda_contiguous(act_beta_ids,"act_beta_ids");
    check_cuda_contiguous(act_row_norms,"act_row_norms");
    check_cuda_contiguous(act_betas,"act_betas");
    check_cuda_contiguous(weight_codes,"weight_codes");
    check_cuda_contiguous(weight_beta_ids,"weight_beta_ids");
    check_cuda_contiguous(weight_row_norms,"weight_row_norms");
    check_cuda_contiguous(weight_betas,"weight_betas");
    check_same_device(act_codes,act_beta_ids,"act_beta_ids");
    check_same_device(act_codes,act_row_norms,"act_row_norms");
    check_same_device(act_codes,act_betas,"act_betas");
    check_same_device(act_codes,weight_codes,"weight_codes");
    check_same_device(act_codes,weight_beta_ids,"weight_beta_ids");
    check_same_device(act_codes,weight_row_norms,"weight_row_norms");
    check_same_device(act_codes,weight_betas,"weight_betas");
    TORCH_CHECK(act_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_row_norms.scalar_type()==torch::kFloat32&&act_row_norms.numel()==1);
    TORCH_CHECK(act_betas.scalar_type()==torch::kFloat32&&act_betas.numel()==4);
    TORCH_CHECK(weight_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(weight_betas.scalar_type()==torch::kFloat32&&weight_betas.numel()==4);
    TORCH_CHECK(split_k==2||split_k==4||split_k==8,
        "split_k must be 2, 4, or 8");
    c10::cuda::CUDAGuard device_guard(act_codes.device());
    const int64_t n_size=weight_row_norms.numel();
    TORCH_CHECK(n_size>0&&n_size%8==0,"N must be a positive multiple of 8");
    TORCH_CHECK(weight_codes.numel()%n_size==0,"weight payload must divide by N");
    const int64_t k_size=weight_codes.numel()*2/n_size;
    TORCH_CHECK(k_size>0&&k_size%8==0&&k_size<=INT_MAX,
        "K must be a supported multiple of 8");
    TORCH_CHECK((k_size/8)%split_k==0,"K blocks must divide evenly by split_k");
    TORCH_CHECK(act_codes.numel()==k_size/2,"A4 code payload shape mismatch");
    TORCH_CHECK(act_beta_ids.numel()==k_size/32,"A4 beta payload shape mismatch");
    TORCH_CHECK(weight_beta_ids.numel()==n_size*k_size/32,
        "W4 beta payload shape mismatch");
    TORCH_CHECK(n_size<=INT_MAX);
    auto partials=torch::empty({split_k,n_size},weight_row_norms.options());
    auto output=torch::empty({1,n_size},weight_row_norms.options());
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    const dim3 grid(unsigned(n_size/8),unsigned(split_k),1);
    nqrt4_w4a4_gemv_m1_splitk_partial_kernel<<<grid,32,0,stream>>>(
        act_codes.data_ptr<uint8_t>(),act_beta_ids.data_ptr<uint8_t>(),
        act_row_norms.data_ptr<float>(),act_betas.data_ptr<float>(),
        weight_codes.data_ptr<uint8_t>(),weight_beta_ids.data_ptr<uint8_t>(),
        weight_row_norms.data_ptr<float>(),weight_betas.data_ptr<float>(),
        int(k_size),int(n_size),int(split_k),weight_offset,weight_seed,
        rsqrtf(float(k_size)),partials.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 M1 split-K partial launch failed: ",
        cudaGetErrorString(error));
    nqrt4_w4a4_splitk_reduce_kernel<<<unsigned((n_size+255)/256),256,0,stream>>>(
        partials.data_ptr<float>(),int(n_size),int(split_k),output.data_ptr<float>());
    error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 M1 split-K reduce launch failed: ",
        cudaGetErrorString(error));
    return output;
}

torch::Tensor nqrt4_w4a4_gemv_m1_splitk_partials_probe_cuda(
    torch::Tensor act_codes,torch::Tensor act_beta_ids,
    torch::Tensor act_row_norms,torch::Tensor act_betas,
    torch::Tensor weight_codes,torch::Tensor weight_beta_ids,
    torch::Tensor weight_row_norms,torch::Tensor weight_betas,
    int64_t weight_offset,int64_t weight_seed,int64_t split_k) {
    check_cuda_contiguous(act_codes,"act_codes");
    check_cuda_contiguous(act_beta_ids,"act_beta_ids");
    check_cuda_contiguous(act_row_norms,"act_row_norms");
    check_cuda_contiguous(act_betas,"act_betas");
    check_cuda_contiguous(weight_codes,"weight_codes");
    check_cuda_contiguous(weight_beta_ids,"weight_beta_ids");
    check_cuda_contiguous(weight_row_norms,"weight_row_norms");
    check_cuda_contiguous(weight_betas,"weight_betas");
    check_same_device(act_codes,act_beta_ids,"act_beta_ids");
    check_same_device(act_codes,act_row_norms,"act_row_norms");
    check_same_device(act_codes,act_betas,"act_betas");
    check_same_device(act_codes,weight_codes,"weight_codes");
    check_same_device(act_codes,weight_beta_ids,"weight_beta_ids");
    check_same_device(act_codes,weight_row_norms,"weight_row_norms");
    check_same_device(act_codes,weight_betas,"weight_betas");
    TORCH_CHECK(act_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_row_norms.scalar_type()==torch::kFloat32&&act_row_norms.numel()==1);
    TORCH_CHECK(act_betas.scalar_type()==torch::kFloat32&&act_betas.numel()==4);
    TORCH_CHECK(weight_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(weight_betas.scalar_type()==torch::kFloat32&&weight_betas.numel()==4);
    TORCH_CHECK(split_k==2||split_k==4||split_k==8,
        "split_k must be 2, 4, or 8");
    c10::cuda::CUDAGuard device_guard(act_codes.device());
    const int64_t n_size=weight_row_norms.numel();
    TORCH_CHECK(n_size>0&&n_size%8==0,"N must be a positive multiple of 8");
    TORCH_CHECK(weight_codes.numel()%n_size==0,"weight payload must divide by N");
    const int64_t k_size=weight_codes.numel()*2/n_size;
    TORCH_CHECK(k_size>0&&k_size%8==0&&k_size<=INT_MAX,
        "K must be a supported multiple of 8");
    TORCH_CHECK((k_size/8)%split_k==0,"K blocks must divide evenly by split_k");
    TORCH_CHECK(act_codes.numel()==k_size/2,"A4 code payload shape mismatch");
    TORCH_CHECK(act_beta_ids.numel()==k_size/32,"A4 beta payload shape mismatch");
    TORCH_CHECK(weight_beta_ids.numel()==n_size*k_size/32,
        "W4 beta payload shape mismatch");
    auto partials=torch::empty({split_k,n_size},weight_row_norms.options());
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    const dim3 grid(unsigned(n_size/8),unsigned(split_k),1);
    nqrt4_w4a4_gemv_m1_splitk_partial_kernel<<<grid,32,0,stream>>>(
        act_codes.data_ptr<uint8_t>(),act_beta_ids.data_ptr<uint8_t>(),
        act_row_norms.data_ptr<float>(),act_betas.data_ptr<float>(),
        weight_codes.data_ptr<uint8_t>(),weight_beta_ids.data_ptr<uint8_t>(),
        weight_row_norms.data_ptr<float>(),weight_betas.data_ptr<float>(),
        int(k_size),int(n_size),int(split_k),weight_offset,weight_seed,
        rsqrtf(float(k_size)),partials.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 M1 split-K partial launch failed: ",
        cudaGetErrorString(error));
    return partials;
}

torch::Tensor nqrt4_w4a4_gemv_m1_splitk_compensated_probe_cuda(
    torch::Tensor act_codes,torch::Tensor act_beta_ids,
    torch::Tensor act_row_norms,torch::Tensor act_betas,
    torch::Tensor weight_codes,torch::Tensor weight_beta_ids,
    torch::Tensor weight_row_norms,torch::Tensor weight_betas,
    int64_t weight_offset,int64_t weight_seed,int64_t split_k) {
    check_cuda_contiguous(act_codes,"act_codes");
    check_cuda_contiguous(act_beta_ids,"act_beta_ids");
    check_cuda_contiguous(act_row_norms,"act_row_norms");
    check_cuda_contiguous(act_betas,"act_betas");
    check_cuda_contiguous(weight_codes,"weight_codes");
    check_cuda_contiguous(weight_beta_ids,"weight_beta_ids");
    check_cuda_contiguous(weight_row_norms,"weight_row_norms");
    check_cuda_contiguous(weight_betas,"weight_betas");
    check_same_device(act_codes,act_beta_ids,"act_beta_ids");
    check_same_device(act_codes,act_row_norms,"act_row_norms");
    check_same_device(act_codes,act_betas,"act_betas");
    check_same_device(act_codes,weight_codes,"weight_codes");
    check_same_device(act_codes,weight_beta_ids,"weight_beta_ids");
    check_same_device(act_codes,weight_row_norms,"weight_row_norms");
    check_same_device(act_codes,weight_betas,"weight_betas");
    TORCH_CHECK(act_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(act_row_norms.scalar_type()==torch::kFloat32&&act_row_norms.numel()==1);
    TORCH_CHECK(act_betas.scalar_type()==torch::kFloat32&&act_betas.numel()==4);
    TORCH_CHECK(weight_codes.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_beta_ids.scalar_type()==torch::kUInt8);
    TORCH_CHECK(weight_row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(weight_betas.scalar_type()==torch::kFloat32&&weight_betas.numel()==4);
    TORCH_CHECK(split_k==2||split_k==4||split_k==8,
        "split_k must be 2, 4, or 8");
    c10::cuda::CUDAGuard device_guard(act_codes.device());
    const int64_t n_size=weight_row_norms.numel();
    TORCH_CHECK(n_size>0&&n_size%8==0,"N must be a positive multiple of 8");
    TORCH_CHECK(weight_codes.numel()%n_size==0,"weight payload must divide by N");
    const int64_t k_size=weight_codes.numel()*2/n_size;
    TORCH_CHECK(k_size>0&&k_size%8==0&&k_size<=INT_MAX,
        "K must be a supported multiple of 8");
    TORCH_CHECK((k_size/8)%split_k==0,"K blocks must divide evenly by split_k");
    TORCH_CHECK(act_codes.numel()==k_size/2,"A4 code payload shape mismatch");
    TORCH_CHECK(act_beta_ids.numel()==k_size/32,"A4 beta payload shape mismatch");
    TORCH_CHECK(weight_beta_ids.numel()==n_size*k_size/32,
        "W4 beta payload shape mismatch");
    TORCH_CHECK(n_size<=INT_MAX);
    auto partials=torch::empty({split_k,n_size},weight_row_norms.options());
    auto output=torch::empty({1,n_size},weight_row_norms.options());
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    const dim3 grid(unsigned(n_size/8),unsigned(split_k),1);
    nqrt4_w4a4_gemv_m1_splitk_compensated_partial_kernel<<<grid,32,0,stream>>>(
        act_codes.data_ptr<uint8_t>(),act_beta_ids.data_ptr<uint8_t>(),
        act_row_norms.data_ptr<float>(),act_betas.data_ptr<float>(),
        weight_codes.data_ptr<uint8_t>(),weight_beta_ids.data_ptr<uint8_t>(),
        weight_row_norms.data_ptr<float>(),weight_betas.data_ptr<float>(),
        int(k_size),int(n_size),int(split_k),weight_offset,weight_seed,
        rsqrtf(float(k_size)),partials.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 M1 compensated split-K partial launch failed: ",
        cudaGetErrorString(error));
    nqrt4_w4a4_splitk_reduce_kernel<<<unsigned((n_size+255)/256),256,0,stream>>>(
        partials.data_ptr<float>(),int(n_size),int(split_k),output.data_ptr<float>());
    error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 M1 compensated split-K reduce launch failed: ",
        cudaGetErrorString(error));
    return output;
}
