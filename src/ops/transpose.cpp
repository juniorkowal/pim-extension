#include <torch/script.h>


torch::Tensor pim_transpose(const at::Tensor& a) {
    return a.transpose(0, 1);
}
