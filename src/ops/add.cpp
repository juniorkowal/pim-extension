#include <torch/script.h>
#include <vector>
#include "pimblas.h"
#include "logging.h"

// torch::Tensor pim_add(const at::Tensor& a, const at::Tensor& b) {
//     std::vector<int64_t> broadcasted_shape;
//     try {
//         broadcasted_shape = at::infer_size(a.sizes(), b.sizes());
//     } catch (const std::exception& e) {
//         show_info("Error: Shapes cannot be broadcasted together.");
//         throw;
//     }

//     if (a.numel() == 1 || b.numel() == 1) { // IF ANY OF THE TENSORS HAS ONLY ONE ELEMENT
//         show_info("One of the tensors is a scalar. Using default torch::add.");
//         return torch::add(a, b);
//     }

//     if (a.sizes() == b.sizes()) { // SAME SIZES
//         //return kernel_add(a, b);
//         show_info("Using vec_add_f for same size tensors ...");
//         torch::Tensor result = torch::zeros_like(a);
        
//         vec_add_f(a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), result.numel());
//         show_info("vec_add_f success ...");
//         return result;
//     }

//     // NOT SAME SIZES
//     if (a.sizes() != b.sizes()) {
//         show_info("Sizes not the same, using torch::add");
//         //torch::Tensor result = torch::zeros_like(a);

//         //vec_add_f(a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), a.numel());
//         //return result;
//         return torch::add(a,b);
//     }
//     show_info("Unexpected tensor shapes behaviour, using torch::add");
//     return torch::add(a, b);
// }

torch::Tensor pim_add(const at::Tensor& a, const at::Tensor& b) {
    // TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
    //            "Only float32 tensors supported");

    // compute broadcasted shape
    auto common_shape = at::infer_size(a.sizes(), b.sizes());
    torch::Tensor result = at::empty(common_shape, a.options());

    // contiguous copies of inputs
    torch::Tensor a_contig = a.expand(common_shape).contiguous();
    torch::Tensor b_contig = b.expand(common_shape).contiguous();

    show_info("Using vec_add_f for same size tensors ...");
    vec_add_f(
        a_contig.data_ptr<float>(),
        b_contig.data_ptr<float>(),
        result.data_ptr<float>(),
        result.numel()
    );
    show_info("vec_add_f success ...");
    return result;
}
