#include <ATen/Tensor.h>

#include "torch_pim/csrc/_logging/Logger.h"
#include "torch_pim/csrc/core/PIMAllocator.cpp"


at::Tensor pim_empty_memory_format(at::IntArrayRef size, 
                                        c10::optional<at::ScalarType> dtype, 
                                        c10::optional<at::Layout> layout, 
                                        c10::optional<at::Device> device, 
                                        c10::optional<bool> pin_memory, 
                                        c10::optional<at::MemoryFormat> memory_format) {
    const at::OptionalDeviceGuard device_guard(device);
    show_info("Custom aten::empty.memory_format() called!");
    constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
    return at::detail::empty_generic(size, &global_pim_alloc, private_use_ks, c10::dtype_or_default(dtype), memory_format);
}

at::Tensor pim_empty_strided(at::IntArrayRef size, 
                                    at::IntArrayRef stride, 
                                    c10::optional<c10::ScalarType> dtype, 
                                    c10::optional<at::Layout> layout, 
                                    c10::optional<at::Device> device, 
                                    c10::optional<bool> pin_memory) {
    const at::OptionalDeviceGuard device_guard(device);
    show_info("Custom aten::empty_strided() called!");
    constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
    return at::detail::empty_strided_generic(size, stride, &global_pim_alloc, private_use_ks, c10::dtype_or_default(dtype));
}

at::Tensor & pim_fill_scalar(at::Tensor & self, const at::Scalar & value) {
    const at::OptionalDeviceGuard device_guard(at::device_of(self));
    // Should fill the tensor's data with "value".
    return self;
}
