#include <climits>

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>

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

template <int MSize>
__global__ void nqrt4_w4a4_cute_large_m_kernel(
    const uint8_t* act_codes,const uint8_t* act_beta_ids,
    const float* act_row_norms,const float* act_betas,
    const uint8_t* weight_codes,const uint8_t* weight_beta_ids,
    const float* weight_row_norms,const float* weight_betas,
    int k_size,int n_size,int64_t weight_offset,int64_t weight_seed,
    float inverse_sqrt_feature,float* output) {
    using namespace cute;
    __shared__ unsigned weight_storage[4*32];
    auto weight_layout=make_layout(
        make_shape(Int<4>{},Int<32>{}),LayoutRight{});
    Tensor weight_fragment=make_tensor(
        make_smem_ptr(weight_storage),weight_layout);
    auto token_layout=make_layout(
        make_shape(Int<MSize/8>{},Int<8>{}),LayoutRight{});
    auto output_layout=make_layout(
        make_shape(Int<MSize>{},n_size),make_stride(n_size,Int<1>{}));
    Tensor output_tensor=make_tensor(make_gmem_ptr(output),output_layout);

    const unsigned warp=threadIdx.x>>5;
    const unsigned lane=threadIdx.x&31;
    const unsigned group=lane>>2,thread=lane&3;
    const int token=int(token_layout(warp,group));
    const int output_base=int(blockIdx.x)*16;
    const int output0=output_base+int(group),output1=output0+8;
    const int blocks_per_row=k_size/8;
    float c0=0,c1=0,c2=0,c3=0;
    for(int tile=0;tile<k_size/16;tile++){
        if(warp==0){
            const int weight_block0=output0*blocks_per_row+tile*2;
            const int weight_block1=output1*blocks_per_row+tile*2;
            weight_fragment(Int<0>{},lane)=nqrt4_decode_block_4lane(
                weight_codes,weight_beta_ids,weight_betas,
                weight_row_norms[output0],weight_block0,
                weight_offset+weight_block0,weight_seed,inverse_sqrt_feature);
            weight_fragment(Int<1>{},lane)=nqrt4_decode_block_4lane(
                weight_codes,weight_beta_ids,weight_betas,
                weight_row_norms[output1],weight_block1,
                weight_offset+weight_block1,weight_seed,inverse_sqrt_feature);
            weight_fragment(Int<2>{},lane)=nqrt4_decode_block_4lane(
                weight_codes,weight_beta_ids,weight_betas,
                weight_row_norms[output0],weight_block0+1,
                weight_offset+weight_block0+1,weight_seed,inverse_sqrt_feature);
            weight_fragment(Int<3>{},lane)=nqrt4_decode_block_4lane(
                weight_codes,weight_beta_ids,weight_betas,
                weight_row_norms[output1],weight_block1+1,
                weight_offset+weight_block1+1,weight_seed,inverse_sqrt_feature);
        }
        __syncthreads();
        const unsigned a0=weight_fragment(Int<0>{},lane);
        const unsigned a1=weight_fragment(Int<1>{},lane);
        const unsigned a2=weight_fragment(Int<2>{},lane);
        const unsigned a3=weight_fragment(Int<3>{},lane);
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
    const int token0=int(token_layout(warp,thread*2));
    output_tensor(token0,output0)=c0;
    output_tensor(token0,output1)=c2;
    output_tensor(token0+1,output0)=c1;
    output_tensor(token0+1,output1)=c3;
}

torch::Tensor nqrt4_w4a4_cute_m128_cuda(
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
    const auto launch=[&](auto m_constant){
        constexpr int M=decltype(m_constant)::value;
        nqrt4_w4a4_cute_large_m_kernel<M><<<unsigned(n_size/16),M*4,0,stream>>>(
            act_codes.data_ptr<uint8_t>(),act_beta_ids.data_ptr<uint8_t>(),
            act_row_norms.data_ptr<float>(),act_betas.data_ptr<float>(),
            weight_codes.data_ptr<uint8_t>(),weight_beta_ids.data_ptr<uint8_t>(),
            weight_row_norms.data_ptr<float>(),weight_betas.data_ptr<float>(),
            int(k_size),int(n_size),weight_offset,weight_seed,rsqrtf(float(k_size)),
            output.data_ptr<float>());
    };
    if(m_size==32)launch(cute::Int<32>{});
    else if(m_size==64)launch(cute::Int<64>{});
    else launch(cute::Int<128>{});
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"W4A4 CuTe M128 launch failed: ",cudaGetErrorString(error));
    return output;
}
