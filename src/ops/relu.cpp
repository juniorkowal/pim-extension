#include <torch/script.h>
#include <vector>
#include "pimblas.h"
#include "logging.h"


at::Tensor relu(const at::Tensor& self) {
    torch::Tensor output = torch::zeros_like(self);
    show_info("Invoking relu_f ...");
    relu_f(self.data_ptr<float>(), output.data_ptr<float>(), self.numel());
    show_info("relu_f success ...");
    return output;
}
