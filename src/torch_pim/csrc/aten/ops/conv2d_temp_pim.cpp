#include <ATen/ATen.h>

#include "src/torch_pim/csrc/_logging/Logger.h"

namespace pim {

at::Tensor convolution(const at::Tensor& input,
                    const at::Tensor& weight,
                    const std::optional<at::Tensor>& bias,
                    const c10::IntArrayRef stride,
                    const c10::IntArrayRef padding,
                    const c10::IntArrayRef dilation,
                    const bool transposed,
                    const c10::IntArrayRef output_padding,
                    const int64_t groups) {

    const auto orig_device = input.device();
    show_info("Temporary conv ...");
    
    at::Tensor input_cpu = input.cpu().to(input.dtype());
    at::Tensor weight_cpu = weight.cpu().to(weight.dtype());
    
    std::optional<at::Tensor> bias_cpu;
    if (bias.has_value() && bias.value().defined()) {
            auto bias_data = bias.value().data_ptr<float>();
            auto bias_numel = bias.value().numel();
            
            auto bias_options = at::TensorOptions()
            .dtype(at::kFloat)
            .device(at::kCPU);
            bias_cpu = at::empty(bias.value().sizes(), bias_options);
            
            if (bias_data != nullptr) {
            auto bias_cpu_data = bias_cpu->data_ptr<float>();
            std::memcpy(bias_cpu_data, bias_data, bias_numel * sizeof(float));
            }
    }

    at::Tensor output = at::convolution(
            input_cpu, 
            weight_cpu, 
            bias_cpu,
            // bias, 
            stride, 
            padding, 
            dilation, 
            transposed, 
            output_padding, 
            groups
    );
    show_info("Temporary conv success ...");
    return output.contiguous().to(orig_device);
}

}