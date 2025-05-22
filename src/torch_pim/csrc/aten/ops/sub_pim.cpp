#include <ATen/ATen.h>

#include "pimblas.h"
#include "src/torch_pim/csrc/_logging/Logger.h"


namespace pim {

at::Tensor& sub(const at::Tensor& self, const at::Tensor& other, const c10::Scalar& alpha, at::Tensor& out) {
    if (self.scalar_type() != at::kFloat && other.scalar_type() != at::kFloat) { // for scalars: add_.Scalar?
        show_info("[sub operator] Falling back to CPU for non-float inputs...");
        at::Tensor cpu_out = at::sub(self.cpu(), other.cpu(), alpha);
        out.copy_(cpu_out.to(out.device()));
        return out;
    }
    auto common_shape = at::infer_size(self.sizes(), other.sizes());

    at::Tensor a_contig = self.expand(common_shape).contiguous();
    at::Tensor b_contig = other.expand(common_shape).contiguous();

    show_info("Using vec_sub_f for same size tensors ...");
    vec_sub_f(
        a_contig.data_ptr<float>(),
        b_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        out.numel()
    );
    show_info("vec_sub_f success ...");
    return out;
}

}
