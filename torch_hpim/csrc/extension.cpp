#include <torch/script.h>
#include <torch/extension.h>
#include "torch_hpim/csrc/logging/logger.h"


#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/extension.h>

#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/view.h>

#include <unordered_map>
// c10::Device get_custom_device();
// void set_custom_device_index(c10::DeviceIndex device_index);
// torch::Tensor privateuse_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
//     c10::optional<at::Layout> layout, c10::optional<at::Device> device,
//     c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format);

// DEVICE STUFF
// generator.cpp
const at::Generator& default_generator(c10::DeviceIndex device_index); // random number generator -> generator.cpp
// allocator.cpp
at::Tensor custom_empty_memory_format(at::IntArrayRef size, // at::tensor vs torch::tensor???
                                        c10::optional<at::ScalarType> dtype, 
                                        c10::optional<at::Layout> layout, 
                                        c10::optional<at::Device> device, 
                                        c10::optional<bool> pin_memory, 
                                        c10::optional<at::MemoryFormat> memory_format);
at::Tensor custom_empty_strided(at::IntArrayRef size, 
                                        at::IntArrayRef stride, 
                                        c10::optional<c10::ScalarType> dtype, 
                                        c10::optional<at::Layout> layout, 
                                        c10::optional<at::Device> device, 
                                        c10::optional<bool> pin_memory);                          
at::Tensor & custom_fill__scalar(at::Tensor& self, const at::Scalar& value);
at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking);
// metadata.cpp
void custom_serialization_registry();
bool check_backend_meta(const at::Tensor& t);
void custom_set_backend_meta(const at::Tensor& t);
// storage.cpp
void custom_storage_registry();
at::Tensor custom__copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst);
at::Tensor& custom_set_source_Storage(at::Tensor& result, c10::Storage src);


// OPERATORS
torch::Tensor pim_mm(const at::Tensor& a, const at::Tensor& b);
torch::Tensor pim_add(const at::Tensor& a, const at::Tensor& b);
torch::Tensor pim_relu(const at::Tensor& a, const c10::optional<at::Tensor>& out = c10::nullopt);


void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    show_info("UPMEM fallback: Operator '" << op.schema().operator_name() << "' not supported. Switching to CPU.");
    at::native::cpu_fallback(op, stack);
}


TORCH_LIBRARY(hpim, m) {
    m.def("pim_add(Tensor a, Tensor b) -> Tensor");
    m.def("pim_mm(Tensor a, Tensor b) -> Tensor");
    m.def("pim_relu(Tensor a, Tensor? out=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // m.impl("empty.memory_format", privateuse_empty_memory_format);
    m.impl("pim_add", pim_add);
    m.impl("pim_mm", pim_mm);
    m.impl("pim_relu", pim_relu);
    // m.impl("_foreach_add.List", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>()); fallback for add ops?
    m.impl("empty.memory_format", &custom_empty_memory_format);
    m.impl("empty_strided", &custom_empty_strided);
    m.impl("fill_.Scalar", &custom_fill__scalar);
    m.impl("_copy_from", &custom__copy_from);
    m.impl("_copy_from_and_resize", &custom__copy_from_and_resize);
    m.impl("set_.source_Storage", &custom_set_source_Storage);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("custom_device", &get_custom_device, "get custom device object"); // to call the function: torch_hpim._C.custom_device
    // m.def("set_custom_device_index", &set_custom_device_index, "set custom device index");
    // m.def("custom_storage_registry", &custom_storage_registry, "set custom storageImpl creat method");
    
    // OPERATORS
    m.def("pim_mm", &pim_mm, "PIM mm implementation"); // note: later we will override things like add.Tensor instead
    m.def("pim_add", &pim_add, "PIM add implementation"); // so this section won't be needed
    m.def("pim_relu", &pim_relu, "PIM relu implementation");

    // DEVICE STUFF
    // m.def("default_generator", &default_generator, "default_generator for privateuse1");
    m.def("custom_serialization_registry", &custom_serialization_registry, "register custom serialization function");
    m.def("check_backend_meta", &check_backend_meta, "check if BackendMeta serialization correctly");
    m.def("custom_set_backend_meta", &custom_set_backend_meta, "a fake set tensor BackendMeta function");
    m.def("custom_storage_registry", &custom_storage_registry, "set custom storageImpl creat method");
}

/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("custom_device", &get_custom_device, "get custom device object");
    // m.def("set_custom_device_index", &set_custom_device_index, "set custom device index");
    m.def("custom_storage_registry", &custom_storage_registry, "set custom storageImpl creat method");
    m.def("custom_set_backend_meta", &custom_set_backend_meta, "a fake set tensor BackendMeta function");
    m.def("check_backend_meta", &check_backend_meta, "check if BackendMeta serialization correctly");
    m.def("custom_serialization_registry", &custom_serialization_registry, "register custom serialization function");
    m.def("default_generator", &default_generator, "default_generator for privateuse1");
}
*/