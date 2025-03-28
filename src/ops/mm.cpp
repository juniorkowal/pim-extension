#include <torch/script.h>
#include <vector>
#include "pimblas.h"
#include "logging.h"


torch::Tensor pim_mm(const at::Tensor& a, const at::Tensor& b) {
    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);
    // TEMPORARY FIX WHEN M IS ODD
    if (m % 2 != 0) {
        show_info("Padding matrix A from " + std::to_string(m) + 
                " to " + std::to_string(m+1) + " rows for odd m");

        auto options = a.options();
        torch::Tensor a_padded = torch::zeros({m+1, k}, options);
        a_padded.narrow(0, 0, m).copy_(a);
        
        torch::Tensor output = torch::zeros({m+1, n}, options);

        const int m_padded = m + 1;
        const float alpha = 1.0f;
        const float beta = 0.0f;

        auto a_contig = a_padded.contiguous();
        auto b_contig = b.contiguous();
        auto out_contig = output.contiguous();

        show_info("Calling gemm with padded matrix...");
        gemm_row_maj_f(&m_padded, &n, &k, &alpha,
                      a_contig.data_ptr<float>(), 
                      b_contig.data_ptr<float>(),
                      &beta, 
                      out_contig.data_ptr<float>());

        show_info("GEMM completed, returning original sized slice");
        return output.narrow(0, 0, m);  // return original size
    }
    // NORMAL CASE (m is even)
    else {
        torch::Tensor output = torch::zeros({m, n}, a.options());

        const float alpha = 1.0f;
        const float beta = 0.0f;

        show_info("Invoking gemm_row_maj_f (normal case)...");
        gemm_row_maj_f(&m, &n, &k, &alpha,
                      a.data_ptr<float>(), b.data_ptr<float>(),
                      &beta, output.data_ptr<float>());
        show_info("gemm_row_maj_f success...");

        return output;
    }
}
