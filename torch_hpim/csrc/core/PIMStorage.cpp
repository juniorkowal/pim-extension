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


// A dummy storageImpl for our custom device, that secretly uses the CPU
c10::intrusive_ptr<c10::StorageImpl> make_custom_storage_impl(c10::StorageImpl::use_byte_size_t,
                                                                c10::SymInt size_bytes,
                                                                c10::DataPtr data_ptr,
                                                                c10::Allocator* allocator,
                                                                bool resizable) {
    c10::intrusive_ptr<c10::StorageImpl> custom_storage_impl;
    if (data_ptr == nullptr) {
        custom_storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(), size_bytes, allocator, resizable);
    } 
    else {
        custom_storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data_ptr), allocator, resizable);
    }
    return custom_storage_impl;
}

// Register our dummy storageImpl create method.
void custom_storage_registry() {
    c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, &make_custom_storage_impl);
}


// Some set operations for the basic use case
at::Tensor& custom_set_source_Storage(at::Tensor& result, c10::Storage src) {
    int64_t new_size = static_cast<int64_t>(src.nbytes() / result.dtype().itemsize());
    c10::IntArrayRef stride = {};
    result.unsafeGetTensorImpl()->set_storage_offset(0);
    at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : std::nullopt;
    at::native::resize_impl_cpu_(result.unsafeGetTensorImpl(),
    new_size, stride_opt,
    /*resize_storage=*/!result.is_meta());
    return result;
}
