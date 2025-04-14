#include <torch/script.h>
#include <torch/extension.h>
#include "logging.h"

c10::Device get_custom_device();
void set_custom_device_index(c10::DeviceIndex device_index);
torch::Tensor privateuse_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
    c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format);


torch::Tensor pim_mm(const at::Tensor& a, const at::Tensor& b);
torch::Tensor pim_add(const at::Tensor& a, const at::Tensor& b);
torch::Tensor pim_relu(const at::Tensor& a, const c10::optional<at::Tensor>& out = c10::nullopt);


TORCH_LIBRARY(hpim, m) {
    m.def("pim_add(Tensor a, Tensor b) -> Tensor");
    m.def("pim_mm(Tensor a, Tensor b) -> Tensor");
    m.def("pim_relu(Tensor a, Tensor? out=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", privateuse_empty_memory_format);
    m.impl("pim_add", pim_add);
    m.impl("pim_mm", pim_mm);
    m.impl("pim_relu", pim_relu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_device", &get_custom_device, "get custom device object");
//    m.def("custom_add_called", &custom_add_called, "check if our custom add function was called");
    m.def("set_custom_device_index", &set_custom_device_index, "set custom device index");
//    m.def("custom_storage_registry", &custom_storage_registry, "set custom storageImpl creat method");
    m.def("pim_mm", &pim_mm, "PIM mm implementation");
    m.def("pim_add", &pim_add, "PIM add implementation");
    m.def("pim_relu", &pim_relu, "PIM relu implementation");
}
//
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    m.def("pim_mm", &pim_mm, "PIM mm implementation");
//    m.def("pim_add", &pim_add, "PIM add implementation");
//    m.def("pim_relu", &pim_relu, "PIM relu implementation");
//}
