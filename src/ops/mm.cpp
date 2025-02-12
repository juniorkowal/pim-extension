#include <torch/script.h>


torch::Tensor mm(const at::Tensor& a, const at::Tensor& b) {
    torch::Tensor result = at::mm(a, b);
    return result;
}