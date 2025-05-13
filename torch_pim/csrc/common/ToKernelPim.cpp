#include <ATen/DeviceGuard.h>
#include <ATen/Tensor.h>

#include "torch_pim/csrc/_logging/Logger.h"


// basic dummy copy_() function, so we can copy from the custom device to/from CPU
at::Tensor pim_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
    const at::OptionalDeviceGuard device_guard(at::device_of(self));
    show_info("Custom aten::_copy_from() called! SELF: " << self.is_cpu() << " DESTINATION: " << dst.is_cpu());
    TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
    TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  
    // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
    TORCH_CHECK(self.sizes() == dst.sizes());
    TORCH_CHECK(self.scalar_type() == dst.scalar_type());
    // show_info("self.is_contiguous: " << self.is_contiguous() << " dst.is_contiguous:" << dst.is_contiguous());
    TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());
  
    std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes());
    return dst;
  }

at::Tensor pim_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    return dst.copy_(self, false);
}
