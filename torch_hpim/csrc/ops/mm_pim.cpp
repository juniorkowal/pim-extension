#include <torch/script.h>
#include <vector>
#include "pimblas.h"
#include "torch_hpim/csrc/_logging/Logger.h"

namespace pim {

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
// torch::Tensor pim_mm(const at::Tensor& a, const at::Tensor& b) {
//    TORCH_CHECK(a.dim() == 2, "Input A must be a 2D matrix");
//    TORCH_CHECK(b.dim() == 2, "Input B must be a 2D matrix");

    int m = self.size(0);  // num of rows in A
    int k = self.size(1);  // num of cols in A (must match rows in B)
    int n = mat2.size(1);  // num of cols in B

    at::Tensor output = at::zeros({m, n}, self.options());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // call sgemm
    show_info("Invoking gemm_row_maj_f ...");
    gemm_row_maj_f(&m, &n, &k, &alpha,
		    self.data_ptr<float>(), mat2.data_ptr<float>(),
                    &beta, output.data_ptr<float>());
    show_info("gemm_row_maj_f success ...");
    return output;
}

}