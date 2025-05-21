#include <torch/script.h>
#include <torch/extension.h>

#include "src/torch_pim/csrc/_logging/Logger.h"


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ device ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
const at::Generator& pim_generator(c10::DeviceIndex device_index);
void pim_serialization_registry();
bool check_backend_meta(const at::Tensor& t);
void pim_set_backend_meta(const at::Tensor& t);
void pim_storage_registry();
at::Tensor& pim_set_source_Storage(at::Tensor& result, c10::Storage src);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ops ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
namespace pim {

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor& add(const at::Tensor& self, const at::Tensor& other, const c10::Scalar& alpha, at::Tensor& out);
at::Tensor relu(const at::Tensor& self);
at::Tensor& relu_(at::Tensor& self);
at::Tensor addmm(const at::Tensor& self,
                    const at::Tensor& mat1,
                    const at::Tensor& mat2,
                    const at::Scalar& beta = 1,
                    const at::Scalar& alpha = 1);
at::Tensor mul(const at::Tensor& self, const at::Tensor& other);
at::Tensor t(const at::Tensor & self); 
at::Tensor convolution(const at::Tensor& input,
                        const at::Tensor& weight,
                        const std::optional<at::Tensor>& bias,
                        const c10::IntArrayRef stride,
                        const c10::IntArrayRef padding,
                        const c10::IntArrayRef dilation,
                        const bool transposed,
                        const c10::IntArrayRef output_padding,
                        const int64_t groups);
at::Tensor& softmax_(const at::Tensor& self,
                    int64_t dim,
                    std::optional<at::ScalarType> dtype,
                    at::Tensor& out);
} // namespace pim

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ common ops ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
at::Tensor pim_reshape(const at::Tensor & self, at::IntArrayRef shape);
at::Tensor pim_empty_memory_format(at::IntArrayRef size,
                                        c10::optional<at::ScalarType> dtype, 
                                        c10::optional<at::Layout> layout, 
                                        c10::optional<at::Device> device, 
                                        c10::optional<bool> pin_memory, 
                                        c10::optional<at::MemoryFormat> memory_format);
at::Tensor pim_empty_strided(at::IntArrayRef size, 
                                    at::IntArrayRef stride, 
                                    c10::optional<c10::ScalarType> dtype, 
                                    c10::optional<at::Layout> layout, 
                                    c10::optional<at::Device> device, 
                                    c10::optional<bool> pin_memory);
at::Tensor & pim_fill_scalar(at::Tensor& self, const at::Scalar& value);
at::Tensor pim_as_strided(const at::Tensor& self,
                                c10::IntArrayRef size,
                                c10::IntArrayRef stride,
                                c10::optional<int64_t> storage_offset_);
at::Tensor pim_view(const at::Tensor& self, c10::IntArrayRef size);
at::Tensor pim_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking);
at::Tensor pim_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst);


TORCH_LIBRARY(torch_pim, m) {
    m.def("add(Tensor self, Tensor other, Scalar alpha, Tensor(a!) out) -> Tensor(a!)");
    m.def("mm(Tensor self, Tensor mat2) -> Tensor");
    m.def("mul(Tensor self, Tensor other) -> Tensor");
    m.def("relu(Tensor self) -> Tensor");
    m.def("relu_(Tensor(a!) self) -> Tensor(a!)");
    m.def("softmax_(Tensor self, int dim, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)");
    m.def("addmm(Tensor self, Tensor mat1, Tensor mat2, Scalar beta=1, Scalar alpha=1) -> Tensor");
    m.def("t(Tensor self) -> Tensor");
    m.def("convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor");
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.out", pim::add);
    m.impl("mm", pim::mm);
    m.impl("mul.Tensor", pim::mul);
    m.impl("relu", pim::relu);
    m.impl("relu_", pim::relu_);
    m.impl("_softmax.out", pim::softmax_);
    m.impl("addmm", pim::addmm);
    m.impl("t", pim::t);
    m.impl("convolution_overrideable", pim::convolution);
    // m.impl("_foreach_add.List", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>()); fallback for add ops?
    m.impl("empty.memory_format", &pim_empty_memory_format);
    m.impl("empty_strided", &pim_empty_strided);
    m.impl("fill_.Scalar", &pim_fill_scalar);
    m.impl("_copy_from", &pim_copy_from);
    m.impl("_copy_from_and_resize", &pim_copy_from_and_resize);
    m.impl("view", &pim_view);
    m.impl("reshape", &pim_reshape);
    m.impl("as_strided", &pim_as_strided);
    m.impl("set_.source_Storage", &pim_set_source_Storage);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ operators ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    m.def("mm", &pim::mm, "PIM mm implementation");
    m.def("mul", &pim::mul, "PIM mul implementation");
    m.def("add.out", &pim::add, "PIM add implementation");
    m.def("relu", &pim::relu, "PIM relu implementation");
    m.def("relu_", &pim::relu_, "PIM relu inplace implementation");
    m.def("_softmax.out", &pim::softmax_, "PIM softmax implementation");
    m.def("addmm", &pim::addmm, "PIM addmm implementation");
    m.def("t", &pim::t, "PIM t (transpose) implementation");
    m.def("convolution_overrideable", &pim::convolution, "PIM convolution temporary implementation");

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ device ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    m.def("default_generator", &pim_generator, "default_generator for privateuse1");
    m.def("custom_serialization_registry", &pim_serialization_registry, "register custom serialization function");
    m.def("check_backend_meta", &check_backend_meta, "check if BackendMeta serialization correctly");
    m.def("custom_set_backend_meta", &pim_set_backend_meta, "a fake set tensor BackendMeta function");
    m.def("custom_storage_registry", &pim_storage_registry, "set custom storageImpl creat method");
}
