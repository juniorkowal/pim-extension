#include <torch/script.h>
#include <torch/extension.h>


torch::Tensor mm(const at::Tensor& a, const at::Tensor& b);
torch::Tensor add(const at::Tensor& a, const at::Tensor& b);
torch::Tensor relu(const at::Tensor& a);


TORCH_LIBRARY(hpim, m) {
    m.def("mm", mm);
    m.def("add", add);
    m.def("relu", relu);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mmm", &mm, "PIM mm implementation");
    m.def("add", &add, "PIM add implementation");
    m.def("relu", &relu, "PIM relu implementation");
}
