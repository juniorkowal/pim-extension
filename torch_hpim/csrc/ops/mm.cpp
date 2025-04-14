#include <torch/script.h>
#include <vector>
#include "pimblas.h"
#include "torch_hpim/csrc/logging/logger.h"


torch::Tensor pim_mm(const at::Tensor& a, const at::Tensor& b) {
//    TORCH_CHECK(a.dim() == 2, "Input A must be a 2D matrix");
//    TORCH_CHECK(b.dim() == 2, "Input B must be a 2D matrix");

    int m = a.size(0);  // num of rows in A
    int k = a.size(1);  // num of cols in A (must match rows in B)
    int n = b.size(1);  // num of cols in B

    torch::Tensor output = torch::zeros({m, n}, a.options());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // call sgemm
    show_info("Invoking gemm_row_maj_f ...");
    gemm_row_maj_f(&m, &n, &k, &alpha,
		    a.data_ptr<float>(), b.data_ptr<float>(),
                    &beta, output.data_ptr<float>());
    show_info("gemm_row_maj_f success ...");
    return output;
}