#include <torch/script.h>
#include <torch/extension.h>
#include "logging.h"


torch::Tensor pim_mm(const at::Tensor& a, const at::Tensor& b);
torch::Tensor pim_add(const at::Tensor& a, const at::Tensor& b);
torch::Tensor pim_relu(const at::Tensor& a, const c10::optional<at::Tensor>& out = c10::nullopt);


TORCH_LIBRARY(hpim, m) {
    m.def("pim_add(Tensor a, Tensor b) -> Tensor");
    m.def("pim_mm(Tensor a, Tensor b) -> Tensor");
    m.def("pim_relu(Tensor a, Tensor? out=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpim, CPU, m) {
    m.impl("pim_add", pim_add);
    m.impl("pim_mm", pim_mm);
    m.impl("pim_relu", pim_relu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pim_mm", &pim_mm, "PIM mm implementation");
    m.def("pim_add", &pim_add, "PIM add implementation");
    m.def("pim_relu", &pim_relu, "PIM relu implementation");
}
