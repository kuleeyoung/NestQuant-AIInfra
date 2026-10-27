#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// Specialized NQRT4 tile: q=14, 4-bit codes, four betas with 2-bit ids.
__global__ void unpack_nqrt4_tile(const uint8_t* code_payload,
                                  const uint8_t* beta_payload,
                                  uint8_t* codes,
                                  uint8_t* beta_ids,
                                  float* cosets) {
    const unsigned lane = threadIdx.x;
    // Four consecutive logical codes per lane: 32 lanes × 4 = 128 codes.
    #pragma unroll
    for (unsigned item = 0; item < 4; ++item) {
        const unsigned index = lane * 4 + item;
        const uint8_t packed = code_payload[index >> 1];
        codes[index] = (packed >> ((index & 1) * 4)) & 0x0f;
    }
    // First 16 lanes unpack one beta id each.
    if (lane < 16) {
        const uint8_t packed = beta_payload[lane >> 2];
        beta_ids[lane] = (packed >> ((lane & 3) * 2)) & 0x03;
        const unsigned base = lane * 8;
        float local[8];
        #pragma unroll
        for (int i=0;i<8;i++) {
            const unsigned index = base + i;
            const uint8_t code_byte = code_payload[index >> 1];
            local[i] = float((code_byte >> ((index & 1) * 4)) & 0x0f);
        }
        const float half_c7 = 0.5f * local[7];
        cosets[base] = 2.0f * local[0] - local[1] + half_c7;
        #pragma unroll
        for (int i=1;i<6;i++) cosets[base+i] = local[i] - local[i+1] + half_c7;
        cosets[base+6] = local[6] + half_c7;
        cosets[base+7] = half_c7;
    }
}

int main() {
    uint8_t expected_codes[128], expected_betas[16];
    uint8_t packed_codes[64] = {}, packed_betas[4] = {};
    for (int i=0;i<128;i++) {
        expected_codes[i] = uint8_t((i*5+3)%14);
        packed_codes[i>>1] |= expected_codes[i] << ((i&1)*4);
    }
    for (int i=0;i<16;i++) {
        expected_betas[i] = uint8_t((i*3+1)%4);
        packed_betas[i>>2] |= expected_betas[i] << ((i&3)*2);
    }
    uint8_t *dpc,*dpb,*dc,*db; float *dcos;
    cudaMalloc(&dpc,64); cudaMalloc(&dpb,4); cudaMalloc(&dc,128); cudaMalloc(&db,16); cudaMalloc(&dcos,128*sizeof(float));
    cudaMemcpy(dpc,packed_codes,64,cudaMemcpyHostToDevice);
    cudaMemcpy(dpb,packed_betas,4,cudaMemcpyHostToDevice);
    unpack_nqrt4_tile<<<1,32>>>(dpc,dpb,dc,db,dcos);
    auto status=cudaDeviceSynchronize();
    uint8_t actual_codes[128], actual_betas[16]; float actual_cosets[128];
    cudaMemcpy(actual_codes,dc,128,cudaMemcpyDeviceToHost);
    cudaMemcpy(actual_betas,db,16,cudaMemcpyDeviceToHost);
    cudaMemcpy(actual_cosets,dcos,128*sizeof(float),cudaMemcpyDeviceToHost);
    int code_errors=0,beta_errors=0,coset_errors=0;
    for(int i=0;i<128;i++) code_errors += actual_codes[i] != expected_codes[i];
    for(int i=0;i<16;i++) beta_errors += actual_betas[i] != expected_betas[i];
    for(int block=0;block<16;block++) {
        const int base=block*8; const float h=0.5f*expected_codes[base+7];
        float ref[8]; ref[0]=2.0f*expected_codes[base]-expected_codes[base+1]+h;
        for(int i=1;i<6;i++) ref[i]=expected_codes[base+i]-expected_codes[base+i+1]+h;
        ref[6]=expected_codes[base+6]+h; ref[7]=h;
        for(int i=0;i<8;i++) coset_errors += actual_cosets[base+i] != ref[i];
    }
    std::printf("status=%d code_errors=%d beta_errors=%d coset_errors=%d first_codes=%u,%u first_betas=%u,%u\n",
                int(status),code_errors,beta_errors,coset_errors,unsigned(actual_codes[0]),unsigned(actual_codes[1]),
                unsigned(actual_betas[0]),unsigned(actual_betas[1]));
    cudaFree(dpc); cudaFree(dpb); cudaFree(dc); cudaFree(db); cudaFree(dcos);
    return status==cudaSuccess && code_errors==0 && beta_errors==0 && coset_errors==0 ? 0 : 1;
}
