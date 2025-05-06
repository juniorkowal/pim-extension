#include <torch/script.h>
#include <vector>
#include "pimblas.h"
#include "torch_hpim/csrc/logging/logger.h"



// torch::Tensor pim_add(const at::Tensor& self, const at::Tensor& other) {
//     at::TensorIteratorConfig config;
//     auto common_shape = at::infer_size(self.sizes(), other.sizes());
//     config.add_owned_output(at::empty(common_shape, self.options()));
//     config.add_owned_input(self);
//     config.add_owned_input(other);
//     auto iter = config.build();
  
//     const bool is_a_contig = iter.input(0).is_contiguous();
//     const bool is_b_contig = iter.input(1).is_contiguous();
    
//     at::Tensor a_contig_tensor = is_a_contig ? iter.input(0) : iter.input(0).contiguous();
//     at::Tensor b_contig_tensor = is_b_contig ? iter.input(1) : iter.input(1).contiguous();
//     at::Tensor result = iter.output(0);
  
//     AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "pim_add", [&]() {
//         const float* a_data = a_contig_tensor.const_data_ptr<float>();
//         const float* b_data = b_contig_tensor.const_data_ptr<float>();
//         float* out_data = result.data_ptr<float>();
//         vec_add_f(
//             a_data, b_data, out_data, a_contig_tensor.numel()
//         );
//     });
  
//     if (!result.is_contiguous()) {
//       result.copy_(result.contiguous());
//     }
  
//     return result;
// }



at::Tensor add(const at::Tensor& self, const at::Tensor& other, const c10::Scalar& alpha=1) {
    // TORCH_CHECK(self.scalar_type() == at::kFloat && other.scalar_type() == at::kFloat,
    //            "Only float32 tensors supported");
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1 && 
                other.device().type() == c10::DeviceType::PrivateUse1,
                "Both tensors must be on PrivateUse1 device");
    auto common_shape = at::infer_size(self.sizes(), other.sizes());
    torch::Tensor result = at::empty(common_shape, self.options());

    torch::Tensor a_contig = self.expand(common_shape).contiguous();
    torch::Tensor b_contig = other.expand(common_shape).contiguous();

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


// torch::Tensor pim_add(const at::Tensor& self, const at::Tensor& other) {
//     std::vector<int64_t> broadcasted_shape;
//     try {
//         broadcasted_shape = at::infer_size(self.sizes(), other.sizes());
//     } catch (const std::exception& e) {
//         show_info("Error: Shapes cannot be broadcasted together.");
//         throw;
//     }

//     if (self.numel() == 1 || other.numel() == 1) { // IF ANY OF THE TENSORS HAS ONLY ONE ELEMENT
//         show_info("One of the tensors is self scalar. Using default torch::add.");
//         return torch::add(self, other);
//     }

//     if (self.sizes() == other.sizes()) { // SAME SIZES
//         //return kernel_add(self, other);
//         show_info("Using vec_add_f for same size tensors ...");
//         torch::Tensor result = torch::zeros_like(self);
        
//         vec_add_f(self.data_ptr<float>(), other.data_ptr<float>(), result.data_ptr<float>(), result.numel());
//         show_info("vec_add_f success ...");
//         return result;
//     }

//     // NOT SAME SIZES
//     if (self.sizes() != other.sizes()) {
//         show_info("Sizes not the same, using torch::add");
//         //torch::Tensor result = torch::zeros_like(self);

//         //vec_add_f(self.data_ptr<float>(), other.data_ptr<float>(), result.data_ptr<float>(), self.numel());
//         //return result;
//         return torch::add(self,other);
//     }
//     show_info("Unexpected tensor shapes behaviour, using torch::add");
//     return torch::add(self, other);
// }