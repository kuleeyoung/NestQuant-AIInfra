#include <cuda_runtime.h>
#include <cstdio>

__global__ void map_blocks(int* packed, long long* global, int blocks_per_row,
                           long long tensor_offset) {
    int output_row=blockIdx.x;
    int k_tile=threadIdx.x;
    int base=output_row*blocks_per_row+k_tile*2;
    int index=(output_row*(blocks_per_row/2)+k_tile)*2;
    packed[index]=base; packed[index+1]=base+1;
    global[index]=tensor_offset+base; global[index+1]=tensor_offset+base+1;
}

int main() {
    constexpr int rows=8,blocks_per_row=16,count=rows*blocks_per_row;
    constexpr long long offset=4096;
    int *dp,host_p[count]; long long *dg,host_g[count];
    cudaMalloc(&dp,sizeof(host_p)); cudaMalloc(&dg,sizeof(host_g));
    map_blocks<<<rows,blocks_per_row/2>>>(dp,dg,blocks_per_row,offset);
    auto status=cudaDeviceSynchronize();
    cudaMemcpy(host_p,dp,sizeof(host_p),cudaMemcpyDeviceToHost);
    cudaMemcpy(host_g,dg,sizeof(host_g),cudaMemcpyDeviceToHost);
    int errors=0;
    for(int row=0;row<rows;row++) for(int block=0;block<blocks_per_row;block++) {
        int index=row*blocks_per_row+block;
        errors += host_p[index]!=index;
        errors += host_g[index]!=offset+index;
    }
    std::printf("status=%d errors=%d row1_first=%d global=%lld last=%d\n",
                int(status),errors,host_p[16],host_g[16],host_p[count-1]);
    cudaFree(dp); cudaFree(dg);
    return status==cudaSuccess && errors==0 ? 0 : 1;
}
