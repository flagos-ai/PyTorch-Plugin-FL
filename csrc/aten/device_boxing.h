// Copyright (c) 2026, BAAI. All rights reserved.
//
// Utilities for boxing/unboxing tensor device metadata between
// flagos (PrivateUse1) and CUDA. Since flagos and CUDA share the same
// GPU memory, temporarily changing device type in tensor metadata allows
// calling native PyTorch CUDA kernels that have device type assertions.

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorImpl.h>

namespace at::native::flagos {

// Change a tensor's device type in-place (metadata only, no data copy).
// Modifies both dispatch key set and DataPtr device.
inline void SetTensorDevice(const at::Tensor& t, c10::DeviceType type) {
  auto* impl = t.unsafeGetTensorImpl();
  auto idx = impl->device().index();
  auto new_device = c10::Device(type, idx);
  impl->_change_backend_component_keys(new_device);
  impl->unsafe_storage().unsafeGetStorageImpl()
      ->_mutable_data_ptr_no_checks().unsafe_set_device(new_device);
}

inline void BoxToCuda(const at::Tensor& t) {
  SetTensorDevice(t, c10::DeviceType::CUDA);
}

inline void UnboxToFlagos(const at::Tensor& t) {
  SetTensorDevice(t, c10::DeviceType::PrivateUse1);
}

// RAII guard: boxes tensors to CUDA, unboxes on destruction.
class DeviceBoxingGuard {
 public:
  template <typename... Tensors>
  explicit DeviceBoxingGuard(const Tensors&... tensors)
      : tensors_{tensors...} {
    for (auto& t : tensors_) {
      if (t.defined()) BoxToCuda(t);
    }
  }
  ~DeviceBoxingGuard() {
    for (auto& t : tensors_) {
      if (t.defined()) UnboxToFlagos(t);
    }
  }
  DeviceBoxingGuard(const DeviceBoxingGuard&) = delete;
  DeviceBoxingGuard& operator=(const DeviceBoxingGuard&) = delete;
 private:
  std::vector<at::Tensor> tensors_;
};

} // namespace at::native::flagos
