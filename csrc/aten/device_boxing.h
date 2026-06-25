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
// Modifies dispatch key set, DataPtr device, and device_opt_.
inline void SetTensorDevice(const at::Tensor& t, c10::DeviceType type) {
  auto* impl = t.unsafeGetTensorImpl();
  auto idx = impl->device().index();
  auto new_device = c10::Device(type, idx);
  impl->_change_backend_component_keys(new_device);
  impl->unsafe_storage().unsafeGetStorageImpl()
      ->_mutable_data_ptr_no_checks().unsafe_set_device(new_device);
  // CRITICAL: Also update device_opt_ so device() returns the new device.
  // device_opt_ is protected, but we can access it via pointer offset.
  // TensorImpl layout: device_opt_ is at a known offset from the base.
  // Safer approach: use reinterpret_cast to access the field directly.
  struct TensorImplAccessor : public c10::TensorImpl {
    void set_device_opt(c10::Device d) { this->device_opt_ = d; }
  };
  static_cast<TensorImplAccessor*>(impl)->set_device_opt(new_device);
}

inline void BoxToCuda(const at::Tensor& t) {
  SetTensorDevice(t, c10::DeviceType::CUDA);
}

inline void UnboxToFlagos(const at::Tensor& t) {
  SetTensorDevice(t, c10::DeviceType::PrivateUse1);
}

// RAII guard: boxes flagos (PrivateUse1) tensors to CUDA, unboxes on destruction.
// CPU/CUDA inputs are left unchanged (e.g. mul/add with a CPU scalar).
class DeviceBoxingGuard {
 public:
  template <typename... Tensors>
  explicit DeviceBoxingGuard(const Tensors&... tensors)
      : tensors_{tensors...} {
    for (auto& t : tensors_) {
      if (t.defined() && t.is_privateuseone()) {
        BoxToCuda(t);
        boxed_.push_back(t);
      }
    }
  }
  ~DeviceBoxingGuard() {
    for (auto& t : boxed_) {
      if (t.defined()) UnboxToFlagos(t);
    }
  }
  DeviceBoxingGuard(const DeviceBoxingGuard&) = delete;
  DeviceBoxingGuard& operator=(const DeviceBoxingGuard&) = delete;
 private:
  std::vector<at::Tensor> tensors_;
  std::vector<at::Tensor> boxed_;
};

// Box/unbox all tensors in a TensorList (for _foreach_* ops).
inline void BoxTensorListToCuda(at::TensorList tensors) {
  for (const auto& t : tensors) {
    if (t.defined() && t.is_privateuseone()) {
      BoxToCuda(t);
    }
  }
}

inline void UnboxTensorListToFlagos(at::TensorList tensors) {
  for (const auto& t : tensors) {
    if (t.defined() && t.device().type() == c10::DeviceType::CUDA) {
      UnboxToFlagos(t);
    }
  }
}

// Box/unbox a vector of Tensors returned by non-inplace _foreach ops.
inline void UnboxTensorVecToFlagos(std::vector<at::Tensor>& tensors) {
  for (auto& t : tensors) {
    if (t.defined() && t.device().type() == c10::DeviceType::CUDA) {
      UnboxToFlagos(t);
    }
  }
}

// RAII guard for TensorList boxing (multiple lists).
// Only unboxes tensors that were actually boxed (originally PrivateUse1),
// leaving genuine CUDA tensors untouched.
class TensorListBoxingGuard {
 public:
  TensorListBoxingGuard() = default;

  void box(at::TensorList tensors) {
    for (const auto& t : tensors) {
      if (t.defined() && t.is_privateuseone()) {
        BoxToCuda(t);
        boxed_.push_back(t);
      }
    }
  }

  ~TensorListBoxingGuard() {
    for (auto& t : boxed_) {
      if (t.defined()) UnboxToFlagos(t);
    }
  }

  TensorListBoxingGuard(const TensorListBoxingGuard&) = delete;
  TensorListBoxingGuard& operator=(const TensorListBoxingGuard&) = delete;

 private:
  std::vector<at::Tensor> boxed_;
};

} // namespace at::native::flagos
