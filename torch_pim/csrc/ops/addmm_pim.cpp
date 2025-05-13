#include <ATen/ATen.h>

#include "pimblas.h"
#include "torch_pim/csrc/_logging/Logger.h"


namespace pim {

at::Tensor addmm(const at::Tensor& self, // bias
                   const at::Tensor& mat1, // input (x)
                   const at::Tensor& mat2, // weight.T
                   const at::Scalar& beta=1, 
                   const at::Scalar& alpha=1) { // should we do it this way, or should we invoke kernels here?

    show_info("Invoking addmm operation ...");
    at::Tensor mm_out = mm(mat1, mat2);

    auto common_shape = at::infer_size(mm_out.sizes(), self.sizes());
    at::Tensor output = at::empty(common_shape, self.options());

    at::Tensor a_contig = self.cpu().expand(common_shape).contiguous();
    at::Tensor b_contig = mm_out.cpu().expand(common_shape).contiguous();

    show_info("Using vec_add_f for same size tensors ...");
    vec_add_f(
        a_contig.data_ptr<float>(),
        b_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        output.numel()
    );
    show_info("vec_add_f success ...");


    show_info("addmm success ...");
    return output;
}

} //namespace pim
