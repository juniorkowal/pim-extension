#include <torch/script.h>
#include <torch/extension.h>
#include "torch_hpim/csrc/_logging/Logger.h"



#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/extension.h>

#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/view.h>

#include <unordered_map>


// DEVICE STUFF
// PIMGenerator.cpp
const at::Generator& default_generator(c10::DeviceIndex device_index); // random number generator -> generator.cpp
// PIMAllocator.cpp
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
at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking);
// PIMSerialization.cpp
void custom_serialization_registry();
bool check_backend_meta(const at::Tensor& t);
void custom_set_backend_meta(const at::Tensor& t);
// PIMStorage.cpp
void custom_storage_registry();
at::Tensor custom__copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst);
at::Tensor& custom_set_source_Storage(at::Tensor& result, c10::Storage src);


// OPERATORS
namespace pim {
at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor add(const at::Tensor& self, const at::Tensor& other, const c10::Scalar& alpha=1);
at::Tensor relu(const at::Tensor& self);
at::Tensor addmm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta = 1,
    const at::Scalar& alpha = 1
);
at::Tensor mul(const at::Tensor& self, const at::Tensor& other);
} // namespace pim

void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    show_info("UPMEM fallback: Operator '" << op.schema().operator_name() << "' not supported. Switching to CPU.");
    at::native::cpu_fallback(op, stack);
}


at::Tensor custom_view(const at::Tensor& self, c10::IntArrayRef size) {
    show_info("Custom view called!");
    auto inferred_size = at::infer_size(size, self.numel());
    auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
    // TORCH_CHECK(
    //     stride.has_value(),
    //     "view size is "
    //     "not compatible with input tensor's size and stride (at least one dimension"
    //     " spans across two contiguous subspaces). Use .reshape(...) instead.", OPS_ERROR(ErrCode::PARAM));
    auto stride_value = *stride;
    auto dst = self;
    // return alias_with_sizes_and_strides_npu(dst, inferred_size, stride_value);
    return at::native::view(self, size);
}

at::Tensor custom_reshape(const at::Tensor & self, at::IntArrayRef shape) {
    return at::_ops::reshape::call(self, c10::fromIntArrayRefSlow(shape));
}

at::Tensor custom_as_strided(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::optional<int64_t> storage_offset_)
{
    auto dst = self;
    // if (InferFormat::IsDefiniteTensorWhenMetaDataChanges(dst, size) && !FormatHelper::IsOpInputBaseFormat(dst)) {
    //     TORCH_WARN_ONCE("current tensor is running as_strided, don't perform inplace operations on the returned value."
    //         " If you encounter this warning and have precision issues,"
    //         " you can try torch.npu.config.allow_internal_format = False to resolve precision issues.")
    //     dst = FormatCastHelper::ApplyBaseFormatTensorBy(dst);
    // }
    auto storage_offset = storage_offset_.value_or(dst.storage_offset());
    auto result = at::detail::make_tensor<at::TensorImpl>(
        c10::TensorImpl::VIEW,
        c10::Storage(dst.storage()),
        dst.key_set(),
        dst.dtype());
    at::native::setStrided(result, size, stride, storage_offset);
    return result;
}



TORCH_LIBRARY(torch_hpim, m) {
    m.def("add(Tensor self, Tensor other, Scalar alpha=1) -> Tensor");
    m.def("mm(Tensor self, Tensor mat2) -> Tensor");
    m.def("mul(Tensor self, Tensor other) -> Tensor");
    m.def("relu(Tensor self) -> Tensor");
    m.def("addmm(Tensor self, Tensor mat1, Tensor mat2, Scalar beta=1, Scalar alpha=1) -> Tensor");
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", pim::add);
    m.impl("mm", pim::mm);
    m.impl("mul.Tensor", pim::mul);
    m.impl("relu", pim::relu);
    m.impl("addmm", pim::addmm);
    // m.impl("_foreach_add.List", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>()); fallback for add ops?
    m.impl("empty.memory_format", &custom_empty_memory_format);
    m.impl("empty_strided", &custom_empty_strided);
    m.impl("fill_.Scalar", &custom_fill__scalar);
    m.impl("_copy_from", &custom__copy_from);
    m.impl("_copy_from_and_resize", &custom__copy_from_and_resize);
    m.impl("set_.source_Storage", &custom_set_source_Storage);
    m.impl("view", &custom_view); //func: view(Tensor(a) self, SymInt[] size) -> Tensor(a)
    m.impl("reshape", &custom_reshape);
    m.impl("as_strided", &custom_as_strided);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m){
    m.fallback(torch::CppFunction::makeFallthrough());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // OPERATORS
    m.def("mm", &pim::mm, "PIM mm implementation");
    m.def("mul", &pim::mul, "PIM mul implementation");
    m.def("add", &pim::add, "PIM add implementation");
    m.def("relu", &pim::relu, "PIM relu implementation");
    m.def("addmm", &pim::addmm, "PIM addmm implementation");

    // DEVICE STUFF
    m.def("default_generator", &default_generator, "default_generator for privateuse1");
    m.def("custom_serialization_registry", &custom_serialization_registry, "register custom serialization function");
    m.def("check_backend_meta", &check_backend_meta, "check if BackendMeta serialization correctly");
    m.def("custom_set_backend_meta", &custom_set_backend_meta, "a fake set tensor BackendMeta function");
    m.def("custom_storage_registry", &custom_storage_registry, "set custom storageImpl creat method");
}
