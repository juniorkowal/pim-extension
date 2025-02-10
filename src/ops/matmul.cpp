#include <torch/script.h>
#include <vector>
#include "pimblas.h"  // Include your PIM kernel headers


torch::Tensor pim_matmul(const at::Tensor& a, const at::Tensor& b) {
//    TORCH_CHECK(a.dim() == 2, "Input A must be a 2D matrix");
//    TORCH_CHECK(b.dim() == 2, "Input B must be a 2D matrix");

    int m = a.size(0);  // Number of rows in A
    int k = a.size(1);  // Number of cols in A (must match rows in B)
    int n = b.size(1);  // Number of cols in B

    torch::Tensor result = torch::zeros({m, n}, a.options());  // output

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // call sgemm
    sgemm_wrapper("N", "N", &m, &n, &k, &alpha,
                  a.data_ptr<float>(), &k,
                  b.data_ptr<float>(), &n,
                  &beta,
                  result.data_ptr<float>(), &n);

    return result;
}
