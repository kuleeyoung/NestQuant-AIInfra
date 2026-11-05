#include <torch/extension.h>

torch::Tensor nqrt4_w4a4_mma_m16_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t);
torch::Tensor nqrt4_w4a4_mma_small_m_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t);
torch::Tensor nqrt4_w4a4_mma_small_m_splitk_probe_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t);
torch::Tensor nqrt4_w4a4_mma_m32_multiwarp_probe_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t);
torch::Tensor nqrt4_w4a4_mma_m128_staged_probe_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t);
torch::Tensor nqrt4_w4a4_mma_m128_pipeline_probe_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t);
torch::Tensor nqrt4_w4a4_gemv_m1_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t);
torch::Tensor nqrt4_w4a4_gemv_m1_legacy_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t);
torch::Tensor nqrt4_w4a4_gemv_m1_splitk_probe_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t);
torch::Tensor nqrt4_w4a4_gemv_m1_splitk_partials_probe_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t);
torch::Tensor nqrt4_w4a4_gemv_m1_splitk_compensated_probe_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t);
torch::Tensor nqa4_decode_m16_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
torch::Tensor nqa4_inspect_m16_cuda(torch::Tensor, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_m16", &nqrt4_w4a4_mma_m16_cuda);
    m.def("run_m8_transposed_probe", &nqrt4_w4a4_mma_small_m_cuda);
    m.def("run_small_m", &nqrt4_w4a4_mma_small_m_cuda);
    m.def("run_small_m_transposed_probe", &nqrt4_w4a4_mma_small_m_cuda);
    m.def("run_small_m_splitk_probe", &nqrt4_w4a4_mma_small_m_splitk_probe_cuda);
    m.def("run_m32_multiwarp_probe", &nqrt4_w4a4_mma_m32_multiwarp_probe_cuda);
    m.def("run_large_m_multiwarp_probe", &nqrt4_w4a4_mma_m32_multiwarp_probe_cuda);
    m.def("run_m128_staged_probe", &nqrt4_w4a4_mma_m128_staged_probe_cuda);
    m.def("run_large_m_staged", &nqrt4_w4a4_mma_m128_staged_probe_cuda);
    m.def("run_m128_pipeline_probe", &nqrt4_w4a4_mma_m128_pipeline_probe_cuda);
    m.def("run_m1", &nqrt4_w4a4_gemv_m1_cuda);
    m.def("run_m1_legacy", &nqrt4_w4a4_gemv_m1_legacy_cuda);
    m.def("run_m1_packed_word_probe", &nqrt4_w4a4_gemv_m1_cuda);
    m.def("run_m1_splitk_probe", &nqrt4_w4a4_gemv_m1_splitk_probe_cuda);
    m.def("run_m1_splitk_partials_probe", &nqrt4_w4a4_gemv_m1_splitk_partials_probe_cuda);
    m.def("run_m1_splitk_compensated_probe", &nqrt4_w4a4_gemv_m1_splitk_compensated_probe_cuda);
    m.def("decode_a4_m16", &nqa4_decode_m16_cuda);
    m.def("inspect_a4_m16", &nqa4_inspect_m16_cuda);
}
