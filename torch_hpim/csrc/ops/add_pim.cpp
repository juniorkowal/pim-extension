#include <torch/script.h>
#include <vector>
#include "pimblas.h"
#include "torch_hpim/csrc/_logging/Logger.h"

namespace pim {

at::Tensor add(const at::Tensor& self, const at::Tensor& other, const c10::Scalar& alpha=1) {
    // TORCH_CHECK(self.scalar_type() == at::kFloat && other.scalar_type() == at::kFloat,
    //            "Only float32 tensors supported");
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1 && 
                other.device().type() == c10::DeviceType::PrivateUse1,
                "Both tensors must be on PrivateUse1 device");
    auto common_shape = at::infer_size(self.sizes(), other.sizes());
    at::Tensor output = at::empty(common_shape, self.options());

    at::Tensor a_contig = self.expand(common_shape).contiguous();
    at::Tensor b_contig = other.expand(common_shape).contiguous();

    show_info("Using vec_add_f for same size tensors ...");
    vec_add_f(
        a_contig.data_ptr<float>(),
        b_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        output.numel()
    );
    show_info("vec_add_f success ...");
    return output;
}

}
