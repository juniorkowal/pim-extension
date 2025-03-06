#include <torch/script.h>
#include <torch/extension.h>
#include "logging.h"

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor add(const at::Tensor& self, const at::Tensor& other, const c10::Scalar& alpha=1);
//torch::Tensor add(const at::Tensor& a, const at::Tensor& b);
at::Tensor relu(const at::Tensor& self);

TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("add.Tensor", add);
    m.impl("mm", mm);         
    m.impl("relu", relu);     
}

TORCH_LIBRARY(hpim, m) {
    m.def("mm", mm);
    m.def("add", add);
    m.def("relu", relu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm", &mm, "PIM mm implementation");
    m.def("add", &add, "PIM add implementation");
    m.def("relu", &relu, "PIM relu implementation");
}
