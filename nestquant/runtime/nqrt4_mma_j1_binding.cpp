#include <torch/extension.h>

torch::Tensor nqrt4_mma_j1_cuda(torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,int64_t,int64_t,double);
torch::Tensor nqrt4_mma_j1_k128_cuda(torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,int64_t,int64_t);
torch::Tensor nqrt4_mma_j1_k128_n64_cuda(torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,int64_t,int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("run",&nqrt4_mma_j1_cuda,"NQRT4 J1 single-tile fused MMA");
    m.def("run_k128",&nqrt4_mma_j1_k128_cuda,"NQRT4 J1 K128 fused MMA");
    m.def("run_k128_n64",&nqrt4_mma_j1_k128_n64_cuda,"NQRT4 J1 K128 N64 fused MMA");
}
