#include <torch/script.h>
#include <vector>
#include "pimblas.h"
#include "logging.h"


torch::Tensor pim_relu(const at::Tensor& a) {
    torch::Tensor output = torch::zeros_like(a);
    show_info("Invoking relu_f ...");
    relu_f(a.data_ptr<float>(), output.data_ptr<float>(), a.numel());
    show_info("relu_f success ...");
    return output;
}
