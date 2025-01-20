#include <torch/script.h>


torch::Tensor pim_add(const at::Tensor& a, const at::Tensor& b) {
    return a + b;
}
