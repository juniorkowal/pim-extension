#include <ATen/Tensor.h>
#include <ATen/native/Resize.h>

#include "src/torch_pim/csrc/_logging/Logger.h"


// A dummy storageImpl for our custom device, that secretly uses the CPU
c10::intrusive_ptr<c10::StorageImpl> make_pim_storage_impl(c10::StorageImpl::use_byte_size_t,
                                                                c10::SymInt size_bytes,
                                                                c10::DataPtr data_ptr,
                                                                c10::Allocator* allocator,
                                                                bool resizable) {
    c10::intrusive_ptr<c10::StorageImpl> pim_storage_impl;
    if (data_ptr == nullptr) {
        pim_storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(), size_bytes, allocator, resizable);
    } 
    else {
        pim_storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data_ptr), allocator, resizable);
    }
    return pim_storage_impl;
}

// Register our dummy storageImpl create method.
void pim_storage_registry() {
    c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, &make_pim_storage_impl);
}


// Some set operations for the basic use case
at::Tensor& pim_set_source_Storage(at::Tensor& result, c10::Storage src) {
    int64_t new_size = static_cast<int64_t>(src.nbytes() / result.dtype().itemsize());
    c10::IntArrayRef stride = {};
    result.unsafeGetTensorImpl()->set_storage_offset(0);
    at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : std::nullopt;
    at::native::resize_impl_cpu_(result.unsafeGetTensorImpl(),
    new_size, stride_opt,
    /*resize_storage=*/!result.is_meta());
    return result;
}
