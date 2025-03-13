#include <torch/script.h>


torch::Tensor relu(const at::Tensor& a) {
    return torch::relu(a);
}
