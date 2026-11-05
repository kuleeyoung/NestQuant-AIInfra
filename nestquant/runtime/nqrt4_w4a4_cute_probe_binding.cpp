#include <torch/extension.h>

torch::Tensor nqrt4_w4a4_cute_m128_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_m128", &nqrt4_w4a4_cute_m128_cuda);
}
