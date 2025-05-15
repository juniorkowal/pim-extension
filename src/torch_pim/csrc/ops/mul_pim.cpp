#include <ATen/ATen.h>

#include "pimblas.h"
#include "src/torch_pim/csrc/_logging/Logger.h"


namespace pim {

    at::Tensor mul(const at::Tensor& self, const at::Tensor& other) {
        at::Tensor output = at::zeros_like(self);
        show_info("Invoking vec_mul_f ...");
        vec_mul_f(self.data_ptr<float>(), other.data_ptr<float>(), output.data_ptr<float>(), self.numel());
        show_info("vec_mul_f success ...");
        return output;
    }
    
} //namespace pim

    