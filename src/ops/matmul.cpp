#include <torch/script.h>
#include <pimblas.h>
#include <iostream>

torch::Tensor pim_matmul(const at::Tensor& a, const at::Tensor& b) {
    torch::Tensor result = at::matmul(a, b);
    std::cout<<pimblas_get_kernel_dir()<<std::endl;
    return result;
}
