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

#include "torch_hpim/csrc/_logging/Logger.h"



// A dummy allocator for our custom device, that secretly uses the CPU
struct DummyCustomAllocator final : at::Allocator {
  DummyCustomAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override { // const override?
    // std::cout << "Custom allocator: " << nbytes << std::endl;
    void *data = c10::alloc_cpu(nbytes); // allocate on cpu for now
    show_info("Custom allocator's allocate() called! Allocate " << nbytes << " at [" << data << "]");
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, 0)}; // just set current device to privateuse:0
  }

  static void ReportAndDelete(void *ptr) {
    if (!ptr) {
      return;
    }
    show_info("Custom allocator's delete() called! Free at [" << ptr << "]");
    c10::free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    default_copy_data(dest, src, count);
  }
};
  
// Register our dummy allocator
static DummyCustomAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);
  

at::Tensor custom_empty_memory_format(at::IntArrayRef size, 
                                      c10::optional<at::ScalarType> dtype, 
                                      c10::optional<at::Layout> layout, 
                                      c10::optional<at::Device> device, 
                                      c10::optional<bool> pin_memory, 
                                      c10::optional<at::MemoryFormat> memory_format) {
  const at::OptionalDeviceGuard device_guard(device);
  show_info("Custom aten::empty.memory_format() called!");
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &global_custom_alloc, private_use_ks, c10::dtype_or_default(dtype), memory_format);
}

at::Tensor custom_empty_strided(at::IntArrayRef size, 
                                at::IntArrayRef stride, 
                                c10::optional<c10::ScalarType> dtype, 
                                c10::optional<at::Layout> layout, 
                                c10::optional<at::Device> device, 
                                c10::optional<bool> pin_memory) {
  const at::OptionalDeviceGuard device_guard(device);
  show_info("Custom aten::empty_strided() called!");
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(size, stride, &global_custom_alloc, private_use_ks, c10::dtype_or_default(dtype));
}

at::Tensor & custom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  // Not bothering to implement.
  // Should fill the tensor's data with "value".
  return self;
}

// basic dummy copy_() function, so we can copy from the custom device to/from CPU
at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  show_info("Custom aten::_copy_from() called! SELF: " << self.is_cpu() << " DESTINATION: " << dst.is_cpu());
  TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");

  // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
  TORCH_CHECK(self.sizes() == dst.sizes());
  TORCH_CHECK(self.scalar_type() == dst.scalar_type());
  TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());

  std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes());
  return dst;
}
