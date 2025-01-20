#include <torch/script.h>


torch::Tensor pim_matmul(const at::Tensor& a, const at::Tensor& b) {
    torch::Tensor result = at::matmul(a, b);
    return result;
}