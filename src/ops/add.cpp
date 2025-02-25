#include <torch/script.h>
#include <vector>
#include "pimblas.h"
#include "logging.h"


torch::Tensor kernel_add(const at::Tensor& a, const at::Tensor& b) {
    show_info("Using kernel_add for tensors of the same shape.");
    size_t last_dim_size = a.size(-1);
    torch::Tensor output = torch::zeros_like(a);
    size_t num_elements = a.numel() / last_dim_size;

    for (size_t i = 0; i < num_elements; ++i) {
        float* a_ptr = a.data_ptr<float>() + i * last_dim_size;
        float* b_ptr = b.data_ptr<float>() + i * last_dim_size;
        float* output_ptr = output.data_ptr<float>() + i * last_dim_size;

        if (vec_add_f(a_ptr, b_ptr, output_ptr, last_dim_size) != 0) {
            show_info("Vector addition kernel failed in a loop.");
            return torch::Tensor();
        }
    }

    show_info("kernel_add element-wise vector addition completed successfully.");
    return output;
}


torch::Tensor add(const at::Tensor& a, const at::Tensor& b) {
    std::vector<int64_t> broadcasted_shape;
    try {
        broadcasted_shape = at::infer_size(a.sizes(), b.sizes());
    } catch (const std::exception& e) {
        show_info("Error: Shapes cannot be broadcasted together.");
        throw;
    }

    if (a.numel() == 1 || b.numel() == 1) { // IF ANY OF THE TENSORS HAS ONLY ONE ELEMENT
        show_info("One of the tensors is a scalar. Using default torch::add.");
        return torch::add(a, b);
    }

    if (a.sizes() == b.sizes()) { // SAME SIZES
        return kernel_add(a, b);
    }

    // NOT SAME SIZES
    if (a.sizes() != b.sizes()) {
        show_info("Sizes not the same, using torch::add.");
        return torch::add(a,b);
    }

    return torch::add(a, b);
}
