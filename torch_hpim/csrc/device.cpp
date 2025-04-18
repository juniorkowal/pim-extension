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
#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <unordered_map>
#include <torch/csrc/utils/pybind.h>

static c10::DeviceIndex custom_device_index = 0;

PyObject* py_factory;
using openreg_ptr_t = uint64_t;


void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}
c10::Device get_custom_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

void set_custom_device_index(c10::DeviceIndex device_index) {
  custom_device_index = device_index;
}

py::function get_method(const char* name) {
  auto factory = py::cast<py::function>(py_factory);
  return factory(name);
}

// A dummy allocator for our custom device, that secretly uses the CPU
struct OpenRegAllocator final : at::Allocator {
  OpenRegAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    std::cout << "OpenReg allocator: " << nbytes << std::endl;
    py::gil_scoped_acquire acquire;
    auto curr_device_idx = get_method("getDevice")().cast<c10::DeviceIndex>();
    auto curr_device =
        c10::Device(c10::DeviceType::PrivateUse1, curr_device_idx);
    void* data = nullptr;
    if (nbytes > 0) {
      data = reinterpret_cast<void*>(
          get_method("malloc")(nbytes).cast<openreg_ptr_t>());
      TORCH_CHECK(
          data, "Failed to allocator ", nbytes, " bytes on openreg device.");
    }
    return {data, data, &ReportAndDelete, curr_device};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr || !Py_IsInitialized()) {
      return;
    }

    py::gil_scoped_acquire acquire;

    PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
    // Always stash, this will be a no-op if there is no error
    PyErr_Fetch(&type, &value, &traceback);

    TORCH_CHECK(
        get_method("free")(reinterpret_cast<openreg_ptr_t>(ptr)).cast<bool>(),
        "Failed to free memory pointer at ",
        ptr);

    // If that user code raised an error, just print it without raising it
    if (PyErr_Occurred()) {
      PyErr_Print();
    }

    // Restore the original error
    PyErr_Restore(type, value, traceback);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    py::gil_scoped_acquire acquire;
    get_method("copy_data")(
        reinterpret_cast<openreg_ptr_t>(dest),
        reinterpret_cast<openreg_ptr_t>(src),
        count);
  }
};

// Register our dummy allocator
static OpenRegAllocator global_openreg_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_openreg_alloc);


torch::Tensor privateuse_empty_memory_format(at::IntArrayRef size, 
                                              c10::optional<at::ScalarType> dtype,
                                              c10::optional<at::Layout> layout, 
                                              c10::optional<at::Device> device,
                                              c10::optional<bool> pin_memory, 
                                              c10::optional<at::MemoryFormat> memory_format) {
  std::cout << "[PrivateUse1] create : size=" << size << std::endl;
  /*
  auto options = at::TensorOptions()
  .dtype(dtype.value_or(at::kFloat))
  .layout(layout.value_or(at::kStrided))
  .device(device.value_or(at::Device("privateuse1")))
  .memory_format(memory_format.value_or(at::MemoryFormat::Contiguous));

  return at::detail::empty_cpu(size, dtype, layout, device, pin_memory, memory_format);
  */
  auto t = at::empty({3}); 
  return t;

}


