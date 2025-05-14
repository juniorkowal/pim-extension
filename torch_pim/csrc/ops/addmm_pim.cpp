#include <ATen/ATen.h>

#include "pimblas.h"
#include "torch_pim/csrc/_logging/Logger.h"


namespace pim {

at::Tensor addmm(const at::Tensor& self, // bias
                   const at::Tensor& mat1, // input (x)
                   const at::Tensor& mat2, // weight.T
                   const at::Scalar& beta=1, 
                   const at::Scalar& alpha=1) {
    const auto orig_device = self.device();
    show_info("Invoking addmm operation ...");
    
    at::Tensor self_contig = self.cpu().contiguous();
    at::Tensor mat1_contig = mat1.cpu().contiguous();
    at::Tensor mat2_contig = mat2.cpu().contiguous();
    int m = mat1_contig.size(0);  // num of rows in mat1
    int k = mat1_contig.size(1);  // num of cols in mat1 (must match rows in mat2)
    int n = mat2_contig.size(1);  // num of cols in mat2
    at::Tensor mm_out = at::zeros({m, n}, self_contig.options());
    
    show_info("Invoking gemm_row_maj_f for matrix multiplication...");
    const float gemm_alpha = 1.0f;
    const float gemm_beta = 0.0f;
    gemm_row_maj_f(&m, &n, &k, &gemm_alpha,
                  mat1_contig.data_ptr<float>(), mat2_contig.data_ptr<float>(),
                  &gemm_beta, mm_out.data_ptr<float>());
    show_info("gemm_row_maj_f success ...");
    
    at::Tensor result = at::empty_like(mm_out);
    
    show_info("Using vec_add_f for adding bias...");
    vec_add_f(
        mm_out.data_ptr<float>(),
        self_contig.data_ptr<float>(),
        result.data_ptr<float>(),
        result.numel()
    );
    show_info("vec_add_f success ...");
    
    show_info("addmm success ...");
    return result.to(orig_device);
}

} //namespace pim
