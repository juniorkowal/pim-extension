#include <ATen/Tensor.h>

#include "src/torch_pim/csrc/_logging/Logger.h"


at::Tensor pim_reshape(const at::Tensor & self, at::IntArrayRef shape) {
    return at::_ops::reshape::call(self, c10::fromIntArrayRefSlow(shape));
}
