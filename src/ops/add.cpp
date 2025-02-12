#include <torch/script.h>


torch::Tensor add(const at::Tensor& a, const at::Tensor& b) {
    return a + b;
}
