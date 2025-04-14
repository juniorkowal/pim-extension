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

static c10::DeviceIndex custom_device_index = 0;

namespace {

} // namespace

namespace at::native {

//REGISTER_PRIVATEUSE1_DISPATCH(abs_stub, &abs_kernel);

} // namespace at::native

void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}
c10::Device get_custom_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

void set_custom_device_index(c10::DeviceIndex device_index) {
  custom_device_index = device_index;
}

torch::Tensor privateuse_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout, c10::optional<at::Device> device,
c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
std::cout << "[PrivateUse1] create : size=" << size << std::endl;
/*
auto options = at::TensorOptions()
.dtype(dtype.value_or(at::kFloat))
.layout(layout.value_or(at::kStrided))
.device(device.value_or(at::Device("privateuse1")))
.memory_format(memory_format.value_or(at::MemoryFormat::Contiguous));

return at::detail::empty_cpu(size, dtype, layout, device, pin_memory, memory_format);
*/
auto t = at::empty({3}); 
return t;

}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("custom_device", &get_custom_device, "get custom device object");
// //    m.def("custom_add_called", &custom_add_called, "check if our custom add function was called");
//     m.def("set_custom_device_index", &set_custom_device_index, "set custom device index");
// //    m.def("custom_storage_registry", &custom_storage_registry, "set custom storageImpl creat method");
// }

// TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
// 	    m.impl("empty.memory_format", [](at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
// 				               c10::optional<at::Layout> layout, c10::optional<at::Device> device,
// 					        c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
// 			            std::cout << "[PrivateUse1] create : size=" << size  std::endl;
//             auto t = at::empty({3}); 
// 	    return t;

// 											        });
// }

