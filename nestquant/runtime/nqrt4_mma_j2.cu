#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "nqrt4_cooperative_device.cuh"

__global__ void nqrt4_mma_j2_k128_n64(
    const __nv_bfloat16* activation,const uint8_t* codes,const uint8_t* beta_ids,
    const float* row_norms,const float* betas,int64_t tensor_offset,int64_t seed,float* output) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3,k=thread*2;
    const int output_base=int(blockIdx.x)*8, output_channel=output_base+int(group);
    float c0=0,c1=0,c2=0,c3=0;
    #pragma unroll
    for(int tile=0;tile<8;tile++){
        const int ak=tile*16;
        unsigned a0=nqrt4_pack_bf16(activation[group*128+ak+k],activation[group*128+ak+k+1]);
        unsigned a1=nqrt4_pack_bf16(activation[(group+8)*128+ak+k],activation[(group+8)*128+ak+k+1]);
        unsigned a2=nqrt4_pack_bf16(activation[group*128+ak+k+8],activation[group*128+ak+k+9]);
        unsigned a3=nqrt4_pack_bf16(activation[(group+8)*128+ak+k+8],activation[(group+8)*128+ak+k+9]);
        const int block=output_channel*16+tile*2;
        unsigned b0=nqrt4_decode_block_4lane(codes,beta_ids,betas,row_norms[output_channel],block,
            tensor_offset+block,seed,0.08838834764831845f);
        unsigned b1=nqrt4_decode_block_4lane(codes,beta_ids,betas,row_norms[output_channel],block+1,
            tensor_offset+block+1,seed,0.08838834764831845f);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    const unsigned column=unsigned(output_base)+thread*2;
    output[group*64+column]=c0;output[group*64+column+1]=c1;
    output[(group+8)*64+column]=c2;output[(group+8)*64+column+1]=c3;
}

torch::Tensor nqrt4_mma_j2_k128_n64_cuda(torch::Tensor activation,torch::Tensor codes,
    torch::Tensor beta_ids,torch::Tensor row_norms,torch::Tensor betas,int64_t offset,int64_t seed){
    TORCH_CHECK(activation.is_cuda()&&activation.scalar_type()==torch::kBFloat16&&activation.sizes()==torch::IntArrayRef({16,128}));
    TORCH_CHECK(codes.is_cuda()&&codes.scalar_type()==torch::kUInt8&&codes.numel()==4096);
    TORCH_CHECK(beta_ids.is_cuda()&&beta_ids.scalar_type()==torch::kUInt8&&beta_ids.numel()==256);
    TORCH_CHECK(row_norms.is_cuda()&&row_norms.scalar_type()==torch::kFloat32&&row_norms.numel()==64);
    TORCH_CHECK(betas.is_cuda()&&betas.scalar_type()==torch::kFloat32&&betas.numel()==4);
    TORCH_CHECK(activation.is_contiguous()&&codes.is_contiguous()&&beta_ids.is_contiguous()&&row_norms.is_contiguous()&&betas.is_contiguous());
    auto output=torch::empty({16,64},activation.options().dtype(torch::kFloat32));
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    nqrt4_mma_j2_k128_n64<<<8,32,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),
        codes.data_ptr<uint8_t>(),beta_ids.data_ptr<uint8_t>(),row_norms.data_ptr<float>(),betas.data_ptr<float>(),offset,seed,output.data_ptr<float>());
    auto error=cudaGetLastError();TORCH_CHECK(error==cudaSuccess,"J2 launch failed: ",cudaGetErrorString(error));
    return output;
}

__global__ void nqrt4_mma_j2_m16_dynamic(
    const __nv_bfloat16* activation,const uint8_t* codes,const uint8_t* beta_ids,
    const float* row_norms,const float* betas,int k_size,int64_t tensor_offset,
    int64_t seed,float inverse_sqrt_feature,float* output) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3,k_pair=thread*2;
    const int output_base=int(blockIdx.x)*8, output_channel=output_base+int(group);
    float c0=0,c1=0,c2=0,c3=0;
    for(int tile=0;tile<k_size/16;tile++){
        const int ak=tile*16;
        unsigned a0=nqrt4_pack_bf16(activation[group*k_size+ak+k_pair],activation[group*k_size+ak+k_pair+1]);
        unsigned a1=nqrt4_pack_bf16(activation[(group+8)*k_size+ak+k_pair],activation[(group+8)*k_size+ak+k_pair+1]);
        unsigned a2=nqrt4_pack_bf16(activation[group*k_size+ak+k_pair+8],activation[group*k_size+ak+k_pair+9]);
        unsigned a3=nqrt4_pack_bf16(activation[(group+8)*k_size+ak+k_pair+8],activation[(group+8)*k_size+ak+k_pair+9]);
        const int blocks_per_row=k_size/8;
        const int block=output_channel*blocks_per_row+tile*2;
        unsigned b0=nqrt4_decode_block_4lane(codes,beta_ids,betas,row_norms[output_channel],block,
            tensor_offset+block,seed,inverse_sqrt_feature);
        unsigned b1=nqrt4_decode_block_4lane(codes,beta_ids,betas,row_norms[output_channel],block+1,
            tensor_offset+block+1,seed,inverse_sqrt_feature);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    const int n_size=int(gridDim.x)*8;
    const unsigned column=unsigned(output_base)+thread*2;
    output[group*n_size+column]=c0;output[group*n_size+column+1]=c1;
    output[(group+8)*n_size+column]=c2;output[(group+8)*n_size+column+1]=c3;
}

torch::Tensor nqrt4_mma_j2_m16_cuda(torch::Tensor activation,torch::Tensor codes,
    torch::Tensor beta_ids,torch::Tensor row_norms,torch::Tensor betas,int64_t offset,int64_t seed){
    TORCH_CHECK(activation.is_cuda()&&activation.scalar_type()==torch::kBFloat16&&activation.dim()==2&&activation.size(0)==16);
    const int64_t k_size=activation.size(1), n_size=row_norms.numel();
    TORCH_CHECK(k_size>0&&k_size%16==0,"K must be a positive multiple of 16");
    TORCH_CHECK(n_size>0&&n_size%8==0,"N must be a positive multiple of 8");
    TORCH_CHECK(codes.is_cuda()&&codes.scalar_type()==torch::kUInt8&&codes.numel()==n_size*k_size/2);
    TORCH_CHECK(beta_ids.is_cuda()&&beta_ids.scalar_type()==torch::kUInt8&&beta_ids.numel()==n_size*k_size/32);
    TORCH_CHECK(row_norms.is_cuda()&&row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(betas.is_cuda()&&betas.scalar_type()==torch::kFloat32&&betas.numel()==4);
    TORCH_CHECK(activation.is_contiguous()&&codes.is_contiguous()&&beta_ids.is_contiguous()&&row_norms.is_contiguous()&&betas.is_contiguous());
    TORCH_CHECK(k_size<=INT_MAX&&n_size<=INT_MAX);
    auto output=torch::empty({16,n_size},activation.options().dtype(torch::kFloat32));
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    nqrt4_mma_j2_m16_dynamic<<<unsigned(n_size/8),32,0,stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),codes.data_ptr<uint8_t>(),
        beta_ids.data_ptr<uint8_t>(),row_norms.data_ptr<float>(),betas.data_ptr<float>(),
        int(k_size),offset,seed,rsqrtf(float(k_size)),output.data_ptr<float>());
    auto error=cudaGetLastError();TORCH_CHECK(error==cudaSuccess,"J2 dynamic launch failed: ",cudaGetErrorString(error));
    return output;
}

__global__ void nqrt4_mma_j2_m16_splitk_partial(
    const __nv_bfloat16* activation,const uint8_t* codes,const uint8_t* beta_ids,
    const float* row_norms,const float* betas,int k_size,int n_size,
    int64_t tensor_offset,int64_t seed,float inverse_sqrt_feature,float* partial) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3,k_pair=thread*2;
    const int output_base=int(blockIdx.x)*8, output_channel=output_base+int(group);
    const int split=int(blockIdx.y), split_count=int(gridDim.y), total_tiles=k_size/16;
    const int tile_begin=(total_tiles*split)/split_count;
    const int tile_end=(total_tiles*(split+1))/split_count;
    float c0=0,c1=0,c2=0,c3=0;
    for(int tile=tile_begin;tile<tile_end;tile++){
        const int ak=tile*16;
        unsigned a0=nqrt4_pack_bf16(activation[group*k_size+ak+k_pair],activation[group*k_size+ak+k_pair+1]);
        unsigned a1=nqrt4_pack_bf16(activation[(group+8)*k_size+ak+k_pair],activation[(group+8)*k_size+ak+k_pair+1]);
        unsigned a2=nqrt4_pack_bf16(activation[group*k_size+ak+k_pair+8],activation[group*k_size+ak+k_pair+9]);
        unsigned a3=nqrt4_pack_bf16(activation[(group+8)*k_size+ak+k_pair+8],activation[(group+8)*k_size+ak+k_pair+9]);
        const int blocks_per_row=k_size/8;
        const int block=output_channel*blocks_per_row+tile*2;
        unsigned b0=nqrt4_decode_block_4lane(codes,beta_ids,betas,row_norms[output_channel],block,
            tensor_offset+block,seed,inverse_sqrt_feature);
        unsigned b1=nqrt4_decode_block_4lane(codes,beta_ids,betas,row_norms[output_channel],block+1,
            tensor_offset+block+1,seed,inverse_sqrt_feature);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    const unsigned column=unsigned(output_base)+thread*2;
    const int split_base=split*16*n_size;
    partial[split_base+group*n_size+column]=c0;
    partial[split_base+group*n_size+column+1]=c1;
    partial[split_base+(group+8)*n_size+column]=c2;
    partial[split_base+(group+8)*n_size+column+1]=c3;
}

__global__ void nqrt4_splitk_reduce(const float* partial,float* output,int elements,int splits){
    const int index=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    if(index>=elements)return;
    float sum=0.0f;
    // Fixed ascending split order makes repeated runs deterministic.
    for(int split=0;split<splits;split++)sum+=partial[split*elements+index];
    output[index]=sum;
}

torch::Tensor nqrt4_mma_j2_m16_splitk_cuda(torch::Tensor activation,torch::Tensor codes,
    torch::Tensor beta_ids,torch::Tensor row_norms,torch::Tensor betas,int64_t offset,
    int64_t seed,int64_t splits){
    TORCH_CHECK(activation.is_cuda()&&activation.scalar_type()==torch::kBFloat16&&activation.dim()==2&&activation.size(0)==16);
    const int64_t k_size=activation.size(1), n_size=row_norms.numel();
    TORCH_CHECK(k_size>0&&k_size%16==0,"K must be a positive multiple of 16");
    TORCH_CHECK(n_size>0&&n_size%8==0,"N must be a positive multiple of 8");
    TORCH_CHECK(splits>0&&splits<=k_size/16,"splits must be in [1, K/16]");
    TORCH_CHECK(codes.is_cuda()&&codes.scalar_type()==torch::kUInt8&&codes.numel()==n_size*k_size/2);
    TORCH_CHECK(beta_ids.is_cuda()&&beta_ids.scalar_type()==torch::kUInt8&&beta_ids.numel()==n_size*k_size/32);
    TORCH_CHECK(row_norms.is_cuda()&&row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(betas.is_cuda()&&betas.scalar_type()==torch::kFloat32&&betas.numel()==4);
    TORCH_CHECK(activation.is_contiguous()&&codes.is_contiguous()&&beta_ids.is_contiguous()&&row_norms.is_contiguous()&&betas.is_contiguous());
    TORCH_CHECK(k_size<=INT_MAX&&n_size<=INT_MAX&&splits<=INT_MAX);
    auto partial=torch::empty({splits,16,n_size},activation.options().dtype(torch::kFloat32));
    auto output=torch::empty({16,n_size},activation.options().dtype(torch::kFloat32));
    dim3 grid(unsigned(n_size/8),unsigned(splits));
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    nqrt4_mma_j2_m16_splitk_partial<<<grid,32,0,stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),codes.data_ptr<uint8_t>(),
        beta_ids.data_ptr<uint8_t>(),row_norms.data_ptr<float>(),betas.data_ptr<float>(),
        int(k_size),int(n_size),offset,seed,rsqrtf(float(k_size)),partial.data_ptr<float>());
    auto error=cudaGetLastError();TORCH_CHECK(error==cudaSuccess,"J2 split-K partial launch failed: ",cudaGetErrorString(error));
    const int elements=int(16*n_size);
    nqrt4_splitk_reduce<<<(elements+255)/256,256,0,stream>>>(partial.data_ptr<float>(),output.data_ptr<float>(),elements,int(splits));
    error=cudaGetLastError();TORCH_CHECK(error==cudaSuccess,"J2 split-K reduction launch failed: ",cudaGetErrorString(error));
    return output;
}

torch::Tensor nqrt4_mma_j2_m16_auto_cuda(torch::Tensor activation,torch::Tensor codes,
    torch::Tensor beta_ids,torch::Tensor row_norms,torch::Tensor betas,int64_t offset,int64_t seed){
    TORCH_CHECK(activation.dim()==2,"activation must be a matrix");
    const int64_t k_size=activation.size(1),n_size=row_norms.numel();
    int64_t splits=16;
    if(k_size>=8192&&n_size<=2048)splits=64;
    else if(n_size>=8192)splits=8;
    splits=min(splits,k_size/16);
    return nqrt4_mma_j2_m16_splitk_cuda(
        activation,codes,beta_ids,row_norms,betas,offset,seed,splits);
}

__global__ void nqrt4_mma_j2_m1_splitk_partial(
    const __nv_bfloat16* activation,const uint8_t* codes,const uint8_t* beta_ids,
    const float* row_norms,const float* betas,int k_size,int n_size,
    int64_t tensor_offset,int64_t seed,float inverse_sqrt_feature,float* partial) {
    const unsigned lane=threadIdx.x,group=lane>>2,thread=lane&3,k_pair=thread*2;
    const int output_base=int(blockIdx.x)*8, output_channel=output_base+int(group);
    const int split=int(blockIdx.y), split_count=int(gridDim.y), total_tiles=k_size/16;
    const int tile_begin=(total_tiles*split)/split_count;
    const int tile_end=(total_tiles*(split+1))/split_count;
    float c0=0,c1=0,c2=0,c3=0;
    for(int tile=tile_begin;tile<tile_end;tile++){
        const int ak=tile*16;
        // MMA still operates on 16 rows.  Only row zero is real; all other A
        // fragments are zero and their output rows are deliberately discarded.
        unsigned a0=0,a1=0,a2=0,a3=0;
        if(group==0){
            a0=nqrt4_pack_bf16(activation[ak+k_pair],activation[ak+k_pair+1]);
            a2=nqrt4_pack_bf16(activation[ak+k_pair+8],activation[ak+k_pair+9]);
        }
        const int blocks_per_row=k_size/8;
        const int block=output_channel*blocks_per_row+tile*2;
        unsigned b0=nqrt4_decode_block_4lane(codes,beta_ids,betas,row_norms[output_channel],block,
            tensor_offset+block,seed,inverse_sqrt_feature);
        unsigned b1=nqrt4_decode_block_4lane(codes,beta_ids,betas,row_norms[output_channel],block+1,
            tensor_offset+block+1,seed,inverse_sqrt_feature);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    if(group==0){
        const unsigned column=unsigned(output_base)+thread*2;
        const int split_base=split*n_size;
        partial[split_base+column]=c0;
        partial[split_base+column+1]=c1;
    }
}

torch::Tensor nqrt4_mma_j2_m1_splitk_cuda(torch::Tensor activation,torch::Tensor codes,
    torch::Tensor beta_ids,torch::Tensor row_norms,torch::Tensor betas,int64_t offset,
    int64_t seed,int64_t splits){
    TORCH_CHECK(activation.is_cuda()&&activation.scalar_type()==torch::kBFloat16&&activation.dim()==2&&activation.size(0)==1);
    const int64_t k_size=activation.size(1),n_size=row_norms.numel();
    TORCH_CHECK(k_size>0&&k_size%16==0,"K must be a positive multiple of 16");
    TORCH_CHECK(n_size>0&&n_size%8==0,"N must be a positive multiple of 8");
    TORCH_CHECK(splits>0&&splits<=k_size/16,"splits must be in [1, K/16]");
    TORCH_CHECK(codes.is_cuda()&&codes.scalar_type()==torch::kUInt8&&codes.numel()==n_size*k_size/2);
    TORCH_CHECK(beta_ids.is_cuda()&&beta_ids.scalar_type()==torch::kUInt8&&beta_ids.numel()==n_size*k_size/32);
    TORCH_CHECK(row_norms.is_cuda()&&row_norms.scalar_type()==torch::kFloat32);
    TORCH_CHECK(betas.is_cuda()&&betas.scalar_type()==torch::kFloat32&&betas.numel()==4);
    TORCH_CHECK(activation.is_contiguous()&&codes.is_contiguous()&&beta_ids.is_contiguous()&&row_norms.is_contiguous()&&betas.is_contiguous());
    auto partial=torch::empty({splits,n_size},activation.options().dtype(torch::kFloat32));
    auto output=torch::empty({1,n_size},activation.options().dtype(torch::kFloat32));
    dim3 grid(unsigned(n_size/8),unsigned(splits));
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    nqrt4_mma_j2_m1_splitk_partial<<<grid,32,0,stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),codes.data_ptr<uint8_t>(),
        beta_ids.data_ptr<uint8_t>(),row_norms.data_ptr<float>(),betas.data_ptr<float>(),
        int(k_size),int(n_size),offset,seed,rsqrtf(float(k_size)),partial.data_ptr<float>());
    auto error=cudaGetLastError();TORCH_CHECK(error==cudaSuccess,"J2 M1 partial launch failed: ",cudaGetErrorString(error));
    const int elements=int(n_size);
    nqrt4_splitk_reduce<<<(elements+255)/256,256,0,stream>>>(partial.data_ptr<float>(),output.data_ptr<float>(),elements,int(splits));
    error=cudaGetLastError();TORCH_CHECK(error==cudaSuccess,"J2 M1 reduction launch failed: ",cudaGetErrorString(error));
    return output;
}

torch::Tensor nqrt4_mma_j2_m1_auto_cuda(torch::Tensor activation,torch::Tensor codes,
    torch::Tensor beta_ids,torch::Tensor row_norms,torch::Tensor betas,int64_t offset,int64_t seed){
    TORCH_CHECK(activation.dim()==2,"activation must be a matrix");
    const int64_t k_size=activation.size(1),n_size=row_norms.numel();
    int64_t splits=16;
    if(k_size>=8192&&n_size<=2048)splits=64;
    else if(n_size>=8192)splits=8;
    splits=min(splits,k_size/16);
    return nqrt4_mma_j2_m1_splitk_cuda(
        activation,codes,beta_ids,row_norms,betas,offset,seed,splits);
}
