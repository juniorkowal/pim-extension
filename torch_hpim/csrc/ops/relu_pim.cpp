#include <ATen/ATen.h>

#include "pimblas.h"
#include "torch_hpim/csrc/_logging/Logger.h"


namespace pim {

at::Tensor relu(const at::Tensor& self) {
    at::Tensor output = at::zeros_like(self);
    show_info("Invoking relu_f ...");
    relu_f(self.data_ptr<float>(), output.data_ptr<float>(), self.numel());
    show_info("relu_f success ...");
    return output;
}

}