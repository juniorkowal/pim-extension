// #include <torch/script.h>
// #include <vector>
// #include "pimblas.h"
// #include "torch_hpim/csrc/_logging/Logger.h"


// at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2);
// at::Tensor add(const at::Tensor& self, const at::Tensor& other, const c10::Scalar& alpha=1);

// torch::Tensor addmm(const at::Tensor& self, 
//                    const at::Tensor& mat1, 
//                    const at::Tensor& mat2, 
//                    const at::Scalar& beta=1, 
//                    const at::Scalar& alpha=1) {
//     TORCH_CHECK(self.dim() == 2, "self must be a 2D matrix");
//     TORCH_CHECK(mat1.dim() == 2, "mat1 must be a 2D matrix");
//     TORCH_CHECK(mat2.dim() == 2, "mat2 must be a 2D matrix");
//     TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1 &&
//                 mat1.device().type() == c10::DeviceType::PrivateUse1 &&
//                 mat2.device().type() == c10::DeviceType::PrivateUse1,
//                 "All tensors must be on PrivateUse1 device");

//     torch::Tensor out1 = mm(mat1, mat2).mul(alpha);
//     torch::Tensor result = add(self.mul(beta), out1);
    
//     return result;
// }


// // torch::Tensor addmm(const at::Tensor& self, 
// //                     const at::Tensor& mat1, 
// //                     const at::Tensor& mat2, 
// //                     const at::Scalar & beta=1, 
// //                     const at::Scalar & alpha=1) {
// //     int m = mat1.size(0);  // num of rows in A
// //     int k = mat1.size(1);  // num of cols in A (must match rows in B)
// //     int n = mat2.size(1);  // num of cols in B

// //     torch::Tensor output = torch::zeros({m, n}, mat1.options());

// //     const float sgemm_alpha = 1.0f;
// //     const float sgemm_beta = 0.0f;

// //     // call sgemm
// //     show_info("Invoking gemm_row_maj_f ... [addmm]");
// //     gemm_row_maj_f(&m, &n, &k, &sgemm_alpha,
// //             mat1.data_ptr<float>(), mat2.data_ptr<float>(),
// //                     &sgemm_beta, output.data_ptr<float>());
// //     show_info("gemm_row_maj_f success ... [addmm]");

// // }



// // torch::Tensor pim_add(const at::Tensor& a, const at::Tensor& b) {
// //     TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1 && 
// //                 b.device().type() == c10::DeviceType::PrivateUse1,
// //                 "Both tensors must be on PrivateUse1 device");
// //     auto common_shape = at::infer_size(a.sizes(), b.sizes());
// //     torch::Tensor result = at::empty(common_shape, a.options());

// //     torch::Tensor a_contig = a.expand(common_shape).contiguous();
// //     torch::Tensor b_contig = b.expand(common_shape).contiguous();

// //     show_info("Using vec_add_f for same size tensors ...");
// //     vec_add_f(
// //         a_contig.data_ptr<float>(),
// //         b_contig.data_ptr<float>(),
// //         result.data_ptr<float>(),
// //         result.numel()
// //     );
// //     show_info("vec_add_f success ...");
// //     return result;
// // }



// // at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {

// //     int m = self.size(0);  // num of rows in A
// //     int k = self.size(1);  // num of cols in A (must match rows in B)
// //     int n = mat2.size(1);  // num of cols in B

// //     torch::Tensor output = torch::zeros({m, n}, self.options());

// //     const float alpha = 1.0f;
// //     const float beta = 0.0f;

// //     // call sgemm
// //     show_info("Invoking gemm_row_maj_f ...");
// //     gemm_row_maj_f(&m, &n, &k, &alpha,
// // 		    self.data_ptr<float>(), mat2.data_ptr<float>(),
// //                     &beta, output.data_ptr<float>());
// //     show_info("gemm_row_maj_f success ...");
// //     return output;
// // }