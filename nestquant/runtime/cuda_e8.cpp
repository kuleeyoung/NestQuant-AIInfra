#include <torch/extension.h>

torch::Tensor e8_nearest_cuda(torch::Tensor x);

torch::Tensor e8_nearest(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 8, "x must have shape [N, 8]");
    return e8_nearest_cuda(x.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("e8_nearest", &e8_nearest, "E8 nearest-point projection (CUDA)");
}
