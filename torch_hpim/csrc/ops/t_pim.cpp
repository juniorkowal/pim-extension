#include <torch/script.h>
#include <vector>
#include "pimblas.h"
#include "torch_hpim/csrc/_logging/Logger.h"


namespace pim { // hacky solution so that linear layer works correctly with device
// aten::t(Tensor(a) self) -> Tensor(a)
at::Tensor t(const at::Tensor & self) {
    const auto orig_device = self.device();
    show_info("Transposing tensor (t op)...");
    at::Tensor result = self.cpu().t().contiguous().to(orig_device);
    show_info("Tensor transposed (t op)...");
    return result;
}
    
}