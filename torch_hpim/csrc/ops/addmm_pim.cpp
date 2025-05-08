#include <torch/script.h>
#include <vector>
#include "pimblas.h"
#include "torch_hpim/csrc/_logging/Logger.h"

namespace pim {

at::Tensor addmm(const at::Tensor& self, // bias
                   const at::Tensor& mat1, // input (x)
                   const at::Tensor& mat2, // weight.T
                   const at::Scalar& beta=1, 
                   const at::Scalar& alpha=1) { // should we do it this way, or should we invoke kernels here?

    show_info("Invoking addmm operation ...");
    // show_info("sizes - mat1: [" << mat1.size(0) << ", " << mat1.size(1) 
    // << "], mat2: [" << mat2.size(0) << ", " << mat2.size(1) 
    // << "], self: [" << self.size(0) << "]");
    // at::Tensor out = mm(mat1, mat2);
    // at::Tensor result = add(self, out);
    at::Tensor result = add(self, mm(mat1, mat2));
    show_info("addmm success ...");
    return result;
}

} //namespace pim
