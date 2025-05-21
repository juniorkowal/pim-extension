#include <ATen/ATen.h>

#include "pimblas.h"
#include "src/torch_pim/csrc/_logging/Logger.h"


namespace pim {

at::Tensor& softmax_(
    const at::Tensor& self,
    int64_t dim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  auto memory_format = self.suggest_memory_format();
  at::Tensor converted = dtype.has_value() ? self.toType(dtype.value()) : self;

  if (converted.scalar_type() != at::ScalarType::Float) {
    converted = converted.to(at::ScalarType::Float);
  }

  at::Tensor contiguous_input = converted.contiguous(memory_format);
  at::Tensor contiguous_output = out.is_contiguous(memory_format) 
      ? out 
      : at::empty(out.sizes(), out.options().memory_format(memory_format));

  auto dim_size = contiguous_input.size(dim);
  auto num_elements = contiguous_input.numel();
  auto outer_size = num_elements / dim_size;

  const float* input_ptr = contiguous_input.const_data_ptr<float>();
  float* output_ptr = contiguous_output.data_ptr<float>();

  show_info("Using softmax pimblas kernel ...");
  for (int64_t i = 0; i < outer_size; ++i) {
    const float* slice_in = input_ptr + i * dim_size;
    float* slice_out = output_ptr + i * dim_size;
    softmax(slice_in, slice_out, dim_size);
  }
  show_info("softmax pimblas success ...");

  if (!out.is_contiguous(memory_format)) {
    out.copy_(contiguous_output);
  }

  return out;
}

}