#include <ATen/DeviceGuard.h>

#include "PIMAllocator.h"
#include <c10/core/impl/alloc_cpu.h>


PIMAllocator global_pim_alloc;

at::DataPtr PIMAllocator::allocate(size_t nbytes) {
    void* data = c10::alloc_cpu(nbytes);
    show_info("Custom allocator's allocate() called! Allocate " << nbytes 
              << " at [" << data << "]");
    return {data, data, &ReportAndDelete, 
            at::Device(at::DeviceType::PrivateUse1, 0)};
}

void PIMAllocator::ReportAndDelete(void* ptr) {
    if (!ptr) return;
    show_info("Custom allocator's delete() called! Free at [" << ptr << "]");
    c10::free_cpu(ptr);
}

at::DeleterFnPtr PIMAllocator::raw_deleter() const {
    return &ReportAndDelete;
}

void PIMAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
    default_copy_data(dest, src, count);
}

void register_pim_allocator() {
    static PIMAllocator global_pim_alloc;
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &global_pim_alloc);
}
