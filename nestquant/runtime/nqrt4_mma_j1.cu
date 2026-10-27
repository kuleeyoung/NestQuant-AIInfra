#include <cuda_runtime.h>
#include <torch/extension.h>
#include "nqrt4_device.cuh"

extern "C" __global__ void nqrt4_mma_j1(
    const __nv_bfloat16* activation, const uint8_t* codes,
    const uint8_t* beta_ids, const float* row_norms, const float* betas,
    int64_t block_offset, int64_t seed, float inverse_sqrt_feature, float* output) {
    const unsigned lane=threadIdx.x, group=lane>>2, thread=lane&3;
    const unsigned k=thread*2;
    unsigned a0=nqrt4_pack_bf16(activation[group*16+k],activation[group*16+k+1]);
    unsigned a1=nqrt4_pack_bf16(activation[(group+8)*16+k],activation[(group+8)*16+k+1]);
    unsigned a2=nqrt4_pack_bf16(activation[group*16+k+8],activation[group*16+k+9]);
    unsigned a3=nqrt4_pack_bf16(activation[(group+8)*16+k+8],activation[(group+8)*16+k+9]);
    __nv_bfloat16 first[8],second[8];
    const int first_block=int(group)*2;
    nqrt4_decode_block(codes,beta_ids,betas,row_norms[group],first_block,
                       block_offset+first_block,seed,inverse_sqrt_feature,first);
    nqrt4_decode_block(codes,beta_ids,betas,row_norms[group],first_block+1,
                       block_offset+first_block+1,seed,inverse_sqrt_feature,second);
    unsigned b0=nqrt4_pack_bf16(first[k],first[k+1]);
    unsigned b1=nqrt4_pack_bf16(second[k],second[k+1]);
    float c0=0,c1=0,c2=0,c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    const unsigned column=thread*2;
    output[group*8+column]=c0; output[group*8+column+1]=c1;
    output[(group+8)*8+column]=c2; output[(group+8)*8+column+1]=c3;
}

extern "C" __global__ void nqrt4_mma_j1_k128(
    const __nv_bfloat16* activation,const uint8_t* codes,const uint8_t* beta_ids,
    const float* row_norms,const float* betas,int64_t tensor_offset,int64_t seed,float* output) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3,k=thread*2;
    float c0=0,c1=0,c2=0,c3=0;
    #pragma unroll
    for(int tile=0;tile<8;tile++) {
        const int ak=tile*16;
        unsigned a0=nqrt4_pack_bf16(activation[group*128+ak+k],activation[group*128+ak+k+1]);
        unsigned a1=nqrt4_pack_bf16(activation[(group+8)*128+ak+k],activation[(group+8)*128+ak+k+1]);
        unsigned a2=nqrt4_pack_bf16(activation[group*128+ak+k+8],activation[group*128+ak+k+9]);
        unsigned a3=nqrt4_pack_bf16(activation[(group+8)*128+ak+k+8],activation[(group+8)*128+ak+k+9]);
        const int block=int(group)*16+tile*2;
        __nv_bfloat16 first[8],second[8];
        nqrt4_decode_block(codes,beta_ids,betas,row_norms[group],block,tensor_offset+block,seed,0.08838834764831845f,first);
        nqrt4_decode_block(codes,beta_ids,betas,row_norms[group],block+1,tensor_offset+block+1,seed,0.08838834764831845f,second);
        unsigned b0=nqrt4_pack_bf16(first[k],first[k+1]);
        unsigned b1=nqrt4_pack_bf16(second[k],second[k+1]);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    const unsigned column=thread*2;
    output[group*8+column]=c0; output[group*8+column+1]=c1;
    output[(group+8)*8+column]=c2; output[(group+8)*8+column+1]=c3;
}

extern "C" __global__ void nqrt4_mma_j1_k128_n64(
    const __nv_bfloat16* activation,const uint8_t* codes,const uint8_t* beta_ids,
    const float* row_norms,const float* betas,int64_t tensor_offset,int64_t seed,float* output) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3,k=thread*2;
    const int output_base=int(blockIdx.x)*8;
    const int output_channel=output_base+int(group);
    float c0=0,c1=0,c2=0,c3=0;
    #pragma unroll
    for(int tile=0;tile<8;tile++) {
        const int ak=tile*16;
        unsigned a0=nqrt4_pack_bf16(activation[group*128+ak+k],activation[group*128+ak+k+1]);
        unsigned a1=nqrt4_pack_bf16(activation[(group+8)*128+ak+k],activation[(group+8)*128+ak+k+1]);
        unsigned a2=nqrt4_pack_bf16(activation[group*128+ak+k+8],activation[group*128+ak+k+9]);
        unsigned a3=nqrt4_pack_bf16(activation[(group+8)*128+ak+k+8],activation[(group+8)*128+ak+k+9]);
        const int packed_block=output_channel*16+tile*2;
        __nv_bfloat16 first[8],second[8];
        nqrt4_decode_block(codes,beta_ids,betas,row_norms[output_channel],packed_block,
                           tensor_offset+packed_block,seed,0.08838834764831845f,first);
        nqrt4_decode_block(codes,beta_ids,betas,row_norms[output_channel],packed_block+1,
                           tensor_offset+packed_block+1,seed,0.08838834764831845f,second);
        unsigned b0=nqrt4_pack_bf16(first[k],first[k+1]);
        unsigned b1=nqrt4_pack_bf16(second[k],second[k+1]);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    const unsigned column=unsigned(output_base)+thread*2;
    output[group*64+column]=c0; output[group*64+column+1]=c1;
    output[(group+8)*64+column]=c2; output[(group+8)*64+column+1]=c3;
}

torch::Tensor nqrt4_mma_j1_cuda(torch::Tensor activation, torch::Tensor codes,
    torch::Tensor beta_ids, torch::Tensor row_norms, torch::Tensor betas,
    int64_t block_offset, int64_t seed, double inverse_sqrt_feature) {
    TORCH_CHECK(activation.is_cuda() && activation.scalar_type()==torch::kBFloat16 && activation.sizes()==torch::IntArrayRef({16,16}));
    TORCH_CHECK(codes.is_cuda() && codes.scalar_type()==torch::kUInt8 && codes.numel()==64);
    TORCH_CHECK(beta_ids.is_cuda() && beta_ids.scalar_type()==torch::kUInt8 && beta_ids.numel()==4);
    TORCH_CHECK(row_norms.is_cuda() && row_norms.scalar_type()==torch::kFloat32 && row_norms.numel()==8);
    TORCH_CHECK(betas.is_cuda() && betas.scalar_type()==torch::kFloat32 && betas.numel()==4);
    TORCH_CHECK(activation.is_contiguous() && codes.is_contiguous() && beta_ids.is_contiguous()
                && row_norms.is_contiguous() && betas.is_contiguous(),"all inputs must be contiguous");
    auto output=torch::empty({16,8},activation.options().dtype(torch::kFloat32));
    nqrt4_mma_j1<<<1,32>>>(
        reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),
        codes.data_ptr<uint8_t>(),beta_ids.data_ptr<uint8_t>(),
        row_norms.data_ptr<float>(),betas.data_ptr<float>(),
        block_offset,seed,float(inverse_sqrt_feature),output.data_ptr<float>());
    auto error=cudaGetLastError();
    TORCH_CHECK(error==cudaSuccess,"nqrt4_mma_j1 launch failed: ",cudaGetErrorString(error));
    return output;
}

torch::Tensor nqrt4_mma_j1_k128_cuda(torch::Tensor activation,torch::Tensor codes,
    torch::Tensor beta_ids,torch::Tensor row_norms,torch::Tensor betas,int64_t offset,int64_t seed) {
    TORCH_CHECK(activation.is_cuda() && activation.scalar_type()==torch::kBFloat16 && activation.sizes()==torch::IntArrayRef({16,128}));
    TORCH_CHECK(codes.is_cuda() && codes.scalar_type()==torch::kUInt8 && codes.numel()==512);
    TORCH_CHECK(beta_ids.is_cuda() && beta_ids.scalar_type()==torch::kUInt8 && beta_ids.numel()==32);
    TORCH_CHECK(row_norms.is_cuda() && row_norms.scalar_type()==torch::kFloat32 && row_norms.numel()==8);
    TORCH_CHECK(betas.is_cuda() && betas.scalar_type()==torch::kFloat32 && betas.numel()==4);
    TORCH_CHECK(activation.is_contiguous()&&codes.is_contiguous()&&beta_ids.is_contiguous()&&row_norms.is_contiguous()&&betas.is_contiguous());
    auto output=torch::empty({16,8},activation.options().dtype(torch::kFloat32));
    nqrt4_mma_j1_k128<<<1,32>>>(reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),codes.data_ptr<uint8_t>(),beta_ids.data_ptr<uint8_t>(),row_norms.data_ptr<float>(),betas.data_ptr<float>(),offset,seed,output.data_ptr<float>());
    auto error=cudaGetLastError(); TORCH_CHECK(error==cudaSuccess,"K128 launch failed: ",cudaGetErrorString(error));
    return output;
}

torch::Tensor nqrt4_mma_j1_k128_n64_cuda(torch::Tensor activation,torch::Tensor codes,
    torch::Tensor beta_ids,torch::Tensor row_norms,torch::Tensor betas,int64_t offset,int64_t seed) {
    TORCH_CHECK(activation.is_cuda() && activation.scalar_type()==torch::kBFloat16 && activation.sizes()==torch::IntArrayRef({16,128}));
    TORCH_CHECK(codes.is_cuda() && codes.scalar_type()==torch::kUInt8 && codes.numel()==4096);
    TORCH_CHECK(beta_ids.is_cuda() && beta_ids.scalar_type()==torch::kUInt8 && beta_ids.numel()==256);
    TORCH_CHECK(row_norms.is_cuda() && row_norms.scalar_type()==torch::kFloat32 && row_norms.numel()==64);
    TORCH_CHECK(betas.is_cuda() && betas.scalar_type()==torch::kFloat32 && betas.numel()==4);
    TORCH_CHECK(activation.is_contiguous()&&codes.is_contiguous()&&beta_ids.is_contiguous()&&row_norms.is_contiguous()&&betas.is_contiguous());
    auto output=torch::empty({16,64},activation.options().dtype(torch::kFloat32));
    nqrt4_mma_j1_k128_n64<<<8,32>>>(reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),
        codes.data_ptr<uint8_t>(),beta_ids.data_ptr<uint8_t>(),row_norms.data_ptr<float>(),
        betas.data_ptr<float>(),offset,seed,output.data_ptr<float>());
    auto error=cudaGetLastError(); TORCH_CHECK(error==cudaSuccess,"K128 N64 launch failed: ",cudaGetErrorString(error));
    return output;
}
