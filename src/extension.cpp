#include <torch/script.h>
#include <torch/extension.h>
#include "logging.h"
//#include <spdlog/spdlog.h>


torch::Tensor mm(const at::Tensor& a, const at::Tensor& b);
torch::Tensor add(const at::Tensor& a, const at::Tensor& b);
torch::Tensor relu(const at::Tensor& a);


//void initialize_extension() {
//    initialize_logging();
//    spdlog::info("Extension initialized with logging.");
//}


//void initialize_extension() {
//    initialize_logging();
//    spdlog::info("Extension initialized with logging.");
//}


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

//static auto _ = []() {
//    initialize_extension();
//    return 0;
//}();
