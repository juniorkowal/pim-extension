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
#include "torch_hpim/csrc/core/PIMAllocator.cpp"


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
    // Should fill the tensor's data with "value".
    return self;
}
