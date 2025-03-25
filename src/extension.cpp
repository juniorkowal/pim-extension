#include <torch/script.h>
#include <torch/extension.h>
#include "logging.h"


torch::Tensor pim_mm(const at::Tensor& a, const at::Tensor& b);
torch::Tensor pim_add(const at::Tensor& a, const at::Tensor& b);
torch::Tensor pim_relu(const at::Tensor& a);


TORCH_LIBRARY(hpim, m) {
    m.def("mm", pim_mm);
    m.def("add", pim_add);
    m.def("relu", pim_relu);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pim_mm", &pim_mm, "PIM mm implementation");
    m.def("pim_add", &pim_add, "PIM add implementation");
    m.def("pim_relu", &pim_relu, "PIM relu implementation");
}
