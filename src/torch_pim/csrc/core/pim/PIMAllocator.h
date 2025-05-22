#pragma once

#include <c10/core/Allocator.h>
#include <ATen/core/aten_interned_strings.h>
#include "src/torch_pim/csrc/_logging/Logger.h"


class PIMAllocator final : public at::Allocator {
public:
    PIMAllocator() = default;
    
    at::DataPtr allocate(size_t nbytes) override;
    at::DeleterFnPtr raw_deleter() const override;
    void copy_data(void* dest, const void* src, std::size_t count) const override;

private:
    static void ReportAndDelete(void* ptr);
};

extern PIMAllocator global_pim_alloc;
void register_pim_allocator();
