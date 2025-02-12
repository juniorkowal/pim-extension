#include <torch/script.h>


torch::Tensor transpose(const at::Tensor& a) {
    return a.transpose(0, 1);
}
