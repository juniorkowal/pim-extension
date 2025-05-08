#include <c10/core/impl/InlineDeviceGuard.h>

#include "torch_hpim/csrc/_logging/Logger.h"


static uint16_t CURR_DEVICE = -1;


struct DummyDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;
    DummyDeviceGuardImpl() {}
    explicit DummyDeviceGuardImpl(c10::DeviceType t) {
      TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
    }
    at::DeviceType type() const override {
      return at::DeviceType::PrivateUse1;
    }
    at::Device exchangeDevice(at::Device d) const override {
      TORCH_INTERNAL_ASSERT(d.type() == at::DeviceType::PrivateUse1);
      TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
      at::Device old_device = getDevice();
      if (old_device.index() != d.index()) {
        // "set the active device"
        CURR_DEVICE = d.index();
      }
      return old_device;
    }
    at::Device getDevice() const override {
      return at::Device(at::DeviceType::PrivateUse1, CURR_DEVICE);
    }
    void setDevice(at::Device d) const override {
      TORCH_INTERNAL_ASSERT(d.type() == at::DeviceType::PrivateUse1);
      TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
      at::Device current_device = getDevice();
      if (current_device != d) {
        CURR_DEVICE = d.index();
      }
    }
    void uncheckedSetDevice(at::Device d) const noexcept override {
      auto current_device = getDevice();
      if (current_device != d) {
        CURR_DEVICE = d.index();
      }
    }
    at::Stream getStream(at::Device d) const noexcept override {
      // no-op
      return at::Stream(at::Stream::DEFAULT, d);
    }
    // NB: These do NOT set the current device
    at::Stream exchangeStream(at::Stream) const noexcept override {
      // no-op
      return at::Stream(at::Stream::DEFAULT, at::Device(at::DeviceType::PrivateUse1, CURR_DEVICE));
    }
    at::DeviceIndex deviceCount() const noexcept override {
      // Hardcoding the number of "valid" devices here at 2.
      return 2;
    }
  
    // Event-related functions
    void record(
        void** /*event*/,
        const at::Stream& /*stream*/,
        const at::DeviceIndex /*device_index*/,
        const c10::EventFlag /*flag*/) const override {
      TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.");
    }
    void block(void* /*event*/, const at::Stream& /*stream*/) const override {
      TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
    }
    bool queryEvent(void* /*event*/) const override {
      TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
    }
    void destroyEvent(void* /*event*/, const at::DeviceIndex /*device_index*/)
        const noexcept override {}
  
    // Stream-related functions
    bool queryStream(const at::Stream& /*stream*/) const override {
      return true;
    }
    void synchronizeStream(const at::Stream& /*stream*/) const override {
      // Don't wait for anything.
    }
};
  
struct DummyGuard {
  explicit DummyGuard() = delete;
  explicit DummyGuard(at::DeviceIndex device_index) : guard_(device_index) {}
  explicit DummyGuard(at::Device device) : guard_(device) {}
  DummyGuard(const DummyGuard&) = delete;
  DummyGuard& operator=(const DummyGuard&) = delete;
  DummyGuard(DummyGuard&& other) = delete;
  DummyGuard& operator=(DummyGuard&& other) = delete;

  void set_device(at::Device device) {
    guard_.set_device(device);
  }

  void reset_device(at::Device device) {
    guard_.reset_device(device);
  }

  void set_index(at::DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  at::Device original_device() const {
    return guard_.original_device();
  }

  at::Device current_device() const {
    return guard_.current_device();
  }

  private:
    c10::impl::InlineDeviceGuard<DummyDeviceGuardImpl> guard_;
};


C10_REGISTER_GUARD_IMPL(PrivateUse1, DummyDeviceGuardImpl);
  