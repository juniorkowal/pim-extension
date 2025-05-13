#include <ATen/ATen.h>

#include "pimblas.h"
#include "torch_pim/csrc/_logging/Logger.h"


namespace pim {

at::Tensor addmm(const at::Tensor& self, // bias
                   const at::Tensor& mat1, // input (x)
                   const at::Tensor& mat2, // weight.T
                   const at::Scalar& beta=1, 
                   const at::Scalar& alpha=1) { // should we do it this way, or should we invoke kernels here?
    const auto orig_device = self.device();
    show_info("Invoking addmm operation ...");
    at::Tensor self_contig = self.cpu().contiguous();
    at::Tensor mat1_contig = mat1.cpu().contiguous();
    at::Tensor mat2_contig = mat2.cpu().contiguous();
    at::Tensor mm_out = mm(mat1_contig, mat2_contig).cpu().contiguous();
    at::Tensor result = add(self_contig, mm_out).to(orig_device);
    show_info("addmm success ...");
    return result;
}

} //namespace pim
