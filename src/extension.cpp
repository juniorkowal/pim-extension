#include <torch/script.h>
#include <torch/extension.h>


torch::Tensor pim_matmul(const at::Tensor& a, const at::Tensor& b);
torch::Tensor pim_add(const at::Tensor& a, const at::Tensor& b);
torch::Tensor pim_transpose(const at::Tensor& a);


TORCH_LIBRARY(hpim, m) {
    m.def("pim_matmul", pim_matmul);
    m.def("pim_add", pim_add);
    m.def("pim_transpose", pim_transpose);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pim_matmul", &pim_matmul, "PIM matmul implementation");
    m.def("pim_add", &pim_add, "PIM add implementation");
    m.def("pim_transpose", &pim_transpose, "PIM transpose implementation");
}
