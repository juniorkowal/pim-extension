#include <ATen/Tensor.h>
#include <torch/csrc/jit/serialization/pickler.h>

#include "torch_hpim/csrc/_logging/Logger.h"


struct CustomBackendMetadata : public c10::BackendMeta {
  // for testing this field will mutate when clone() is called by shallow_copy_from.
  int backend_version_format_{-1};
  int format_number_{-1};
  mutable bool cloned_{false};
  // define the constructor
  CustomBackendMetadata(int backend_version_format, int format_number) :
      backend_version_format_(backend_version_format), format_number_(format_number) {}
  c10::intrusive_ptr<c10::BackendMeta> clone(
      const c10::intrusive_ptr<c10::BackendMeta>& ptr) const override {
    cloned_ = true;
    return c10::BackendMeta::clone(ptr);
  }
};

// we need to register two functions for serialization
void for_serialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
  if (t.unsafeGetTensorImpl()->get_backend_meta_intrusive_ptr() == nullptr) {
    return;
  }
  auto tmeta = dynamic_cast<CustomBackendMetadata*>(t.unsafeGetTensorImpl()->get_backend_meta());
  if (tmeta->backend_version_format_ == 1) {
    m["backend_version_format"] = true;
  }
  if (tmeta->format_number_ == 29) {
    m["format_number"] = true;
  }
}

void for_deserialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
  int backend_version_format{-1};
  int format_number{-1};
  if (m.find("backend_version_format") != m.end()) {
    backend_version_format = 1;
  }
  if (m.find("format_number") != m.end()) {
    format_number = 29;
  }
  c10::intrusive_ptr<c10::BackendMeta> new_tmeta{
      std::unique_ptr<c10::BackendMeta>(
      new CustomBackendMetadata(backend_version_format, format_number))
    };
  t.unsafeGetTensorImpl()->set_backend_meta(new_tmeta);
}

void custom_serialization_registry() {
  torch::jit::TensorBackendMetaRegistry(c10::DeviceType::PrivateUse1,
                                        &for_serialization,
                                        &for_deserialization);
}

//check if BackendMeta serialization correctly
bool check_backend_meta(const at::Tensor& t) {
  if (t.unsafeGetTensorImpl()->get_backend_meta_intrusive_ptr()) {
    CustomBackendMetadata* tmeta = dynamic_cast<CustomBackendMetadata*>(t.unsafeGetTensorImpl()->get_backend_meta());
    if (tmeta->backend_version_format_==1 && tmeta->format_number_==29) {
      return true;
    }
  }
  return false;
}

// a fake set function is exposed to the Python side
void custom_set_backend_meta(const at::Tensor& t) {
  int backend_version_format{1};
  int format_number{29};
  c10::intrusive_ptr<c10::BackendMeta> new_tmeta {
      std::unique_ptr<c10::BackendMeta>(
      new CustomBackendMetadata(backend_version_format, format_number))
    };
  t.unsafeGetTensorImpl()->set_backend_meta(new_tmeta);
}