#include <torch/script.h>
#include <torch/extension.h>

#include "torch_pim/csrc/_logging/Logger.h"


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ device ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
const at::Generator& default_generator(c10::DeviceIndex device_index);
void custom_serialization_registry();
bool check_backend_meta(const at::Tensor& t);
void custom_set_backend_meta(const at::Tensor& t);
void custom_storage_registry();
at::Tensor& custom_set_source_Storage(at::Tensor& result, c10::Storage src);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ops ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
namespace pim {

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor add(const at::Tensor& self, const at::Tensor& other, const c10::Scalar& alpha=1);
at::Tensor relu(const at::Tensor& self);
at::Tensor addmm(const at::Tensor& self,
                    const at::Tensor& mat1,
                    const at::Tensor& mat2,
                    const at::Scalar& beta = 1,
                    const at::Scalar& alpha = 1);
at::Tensor mul(const at::Tensor& self, const at::Tensor& other);
at::Tensor t(const at::Tensor & self); 

} // namespace pim

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ common ops ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
at::Tensor custom_reshape(const at::Tensor & self, at::IntArrayRef shape);
at::Tensor custom_empty_memory_format(at::IntArrayRef size,
                                        c10::optional<at::ScalarType> dtype, 
                                        c10::optional<at::Layout> layout, 
                                        c10::optional<at::Device> device, 
                                        c10::optional<bool> pin_memory, 
                                        c10::optional<at::MemoryFormat> memory_format);
at::Tensor custom_empty_strided(at::IntArrayRef size, 
                                    at::IntArrayRef stride, 
                                    c10::optional<c10::ScalarType> dtype, 
                                    c10::optional<at::Layout> layout, 
                                    c10::optional<at::Device> device, 
                                    c10::optional<bool> pin_memory);
at::Tensor & custom_fill__scalar(at::Tensor& self, const at::Scalar& value);
at::Tensor custom_as_strided(const at::Tensor& self,
                                c10::IntArrayRef size,
                                c10::IntArrayRef stride,
                                c10::optional<int64_t> storage_offset_);
at::Tensor custom_view(const at::Tensor& self, c10::IntArrayRef size);
at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking);
at::Tensor custom__copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst);


TORCH_LIBRARY(torch_pim, m) {
    m.def("add(Tensor self, Tensor other, Scalar alpha=1) -> Tensor");
    m.def("mm(Tensor self, Tensor mat2) -> Tensor");
    m.def("mul(Tensor self, Tensor other) -> Tensor");
    m.def("relu(Tensor self) -> Tensor");
    m.def("addmm(Tensor self, Tensor mat1, Tensor mat2, Scalar beta=1, Scalar alpha=1) -> Tensor");
    m.def("t(Tensor self) -> Tensor");
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", pim::add);
    m.impl("mm", pim::mm);
    m.impl("mul.Tensor", pim::mul);
    m.impl("relu", pim::relu);
    m.impl("addmm", pim::addmm);
    m.impl("t", pim::t);
    // m.impl("_foreach_add.List", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>()); fallback for add ops?
    m.impl("empty.memory_format", &custom_empty_memory_format);
    m.impl("empty_strided", &custom_empty_strided);
    m.impl("fill_.Scalar", &custom_fill__scalar);
    m.impl("_copy_from", &custom__copy_from);
    m.impl("_copy_from_and_resize", &custom__copy_from_and_resize);
    m.impl("view", &custom_view);
    m.impl("reshape", &custom_reshape);
    m.impl("as_strided", &custom_as_strided);
    m.impl("set_.source_Storage", &custom_set_source_Storage);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ operators ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    m.def("mm", &pim::mm, "PIM mm implementation");
    m.def("mul", &pim::mul, "PIM mul implementation");
    m.def("add", &pim::add, "PIM add implementation");
    m.def("relu", &pim::relu, "PIM relu implementation");
    m.def("addmm", &pim::addmm, "PIM addmm implementation");
    m.def("t", &pim::t, "PIM t (transpose) implementation");

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ device ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    m.def("default_generator", &default_generator, "default_generator for privateuse1");
    m.def("custom_serialization_registry", &custom_serialization_registry, "register custom serialization function");
    m.def("check_backend_meta", &check_backend_meta, "check if BackendMeta serialization correctly");
    m.def("custom_set_backend_meta", &custom_set_backend_meta, "a fake set tensor BackendMeta function");
    m.def("custom_storage_registry", &custom_storage_registry, "set custom storageImpl creat method");
}
