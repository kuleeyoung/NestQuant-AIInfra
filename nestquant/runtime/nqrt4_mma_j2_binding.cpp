#include <torch/extension.h>
torch::Tensor nqrt4_mma_j2_k128_n64_cuda(torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,int64_t,int64_t);
torch::Tensor nqrt4_mma_j2_m16_cuda(torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,int64_t,int64_t);
torch::Tensor nqrt4_mma_j2_m16_splitk_cuda(torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);
torch::Tensor nqrt4_mma_j2_m16_auto_cuda(torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,int64_t,int64_t);
torch::Tensor nqrt4_mma_j2_m1_splitk_cuda(torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);
torch::Tensor nqrt4_mma_j2_m1_auto_cuda(torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,int64_t,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("run_k128_n64",&nqrt4_mma_j2_k128_n64_cuda);
    m.def("run_m16",&nqrt4_mma_j2_m16_cuda);
    m.def("run_m16_splitk",&nqrt4_mma_j2_m16_splitk_cuda);
    m.def("run_m16_auto",&nqrt4_mma_j2_m16_auto_cuda);
    m.def("run_m1_splitk",&nqrt4_mma_j2_m1_splitk_cuda);
    m.def("run_m1_auto",&nqrt4_mma_j2_m1_auto_cuda);
}
