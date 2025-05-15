#include <ATen/native/Resize.h>
#include <ATen/Tensor.h>
#include <ATen/InferSize.h>

#include "src/torch_pim/csrc/_logging/Logger.h"


at::Tensor pim_as_strided(
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

at::Tensor pim_view(const at::Tensor& self, c10::IntArrayRef size) {
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