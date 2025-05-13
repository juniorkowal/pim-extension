#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/alloc_cpu.h>

#include <ATen/EmptyTensor.h>
#include <ATen/DeviceGuard.h>

#include "torch_pim/csrc/_logging/Logger.h"


// A dummy allocator for our custom device, that secretly uses the CPU
struct PIMAllocator final : at::Allocator {
  PIMAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override { // const override?
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
    // show_info("Dest: " << dest << " Src: " << src << " Count: " << count);
    default_copy_data(dest, src, count);
  }
};
  
// Register our dummy allocator
static PIMAllocator global_pim_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_pim_alloc);