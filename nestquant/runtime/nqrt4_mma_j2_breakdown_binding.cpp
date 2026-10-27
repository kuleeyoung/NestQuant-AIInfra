#include <torch/extension.h>

torch::Tensor nqrt4_mma_j2_breakdown_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &nqrt4_mma_j2_breakdown_cuda);
}
