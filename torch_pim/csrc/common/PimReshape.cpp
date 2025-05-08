#include <ATen/Tensor.h>

#include "torch_pim/csrc/_logging/Logger.h"


at::Tensor custom_reshape(const at::Tensor & self, at::IntArrayRef shape) {
    return at::_ops::reshape::call(self, c10::fromIntArrayRefSlow(shape));
}
