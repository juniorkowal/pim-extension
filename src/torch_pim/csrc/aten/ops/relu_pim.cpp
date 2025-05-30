#include <ATen/ATen.h>

#include "pimblas.h"
#include "src/torch_pim/csrc/_logging/Logger.h"


namespace pim {

at::Tensor relu(const at::Tensor& self) {
    at::Tensor output = at::zeros_like(self);
    show_info("[relu] Invoking relu_f ...");
    relu_f(self.data_ptr<float>(), output.data_ptr<float>(), self.numel());
    show_info("[relu] relu_f success ...");
    return output;
}

at::Tensor& relu_(at::Tensor& self) {
    show_info("[relu_] Invoking relu_f ...");
    relu_f(self.data_ptr<float>(), self.data_ptr<float>(), self.numel());
    show_info("[relu_] relu_f success ...");
    return self;
}

}