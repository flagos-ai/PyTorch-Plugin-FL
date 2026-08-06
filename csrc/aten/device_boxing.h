// Copyright (c) 2026, BAAI. All rights reserved.
//
// Utilities for boxing/unboxing tensor device metadata between
// flagos (PrivateUse1) and CUDA. Since flagos and CUDA share the same
// GPU memory, temporarily changing device type in tensor metadata allows
// calling native PyTorch CUDA kernels that have device type assertions.

#pragma once

#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/SmallVector.h>

#include <type_traits>

namespace at::native::flagos {

// Change a tensor's device type in-place (metadata only, no data copy).
// Modifies dispatch key set, DataPtr device, and device_opt_.
// Operates directly on a TensorImpl* so callers holding a raw impl pointer
// (e.g. the boxing guards, which record impls to avoid refcount churn) don't
// need to reconstruct an owning Tensor to unbox.
inline void SetTensorImplDevice(c10::TensorImpl* impl, c10::DeviceType type) {
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

inline void SetTensorDevice(const at::Tensor& t, c10::DeviceType type) {
  // Undefined tensors (e.g. an unrequested grad in a *_backward output tuple,
  // like the bias grad of convolution_backward when output_mask[2]==false) have
  // no TensorImpl; touching device() would dereference null -> "tensor does not
  // have a device". Nothing to rebox, so skip.
  if (!t.defined()) return;
  SetTensorImplDevice(t.unsafeGetTensorImpl(), type);
}

inline void BoxToCuda(const at::Tensor& t) {
  SetTensorDevice(t, c10::DeviceType::CUDA);
}

inline void UnboxToFlagos(const at::Tensor& t) {
  SetTensorDevice(t, c10::DeviceType::PrivateUse1);
}

// RAII guard: boxes flagos (PrivateUse1) tensors to CUDA, unboxes on destruction.
// CPU/CUDA inputs are left unchanged (e.g. mul/add with a CPU scalar).
//
// Boxed tensors are tracked by raw TensorImpl* rather than an owning Tensor:
// the box target always outlives the guard (callers pass named tensors, never
// temporaries), so no ownership is needed here and we avoid an atomic
// refcount inc/dec per boxed tensor on both box and unbox.
class DeviceBoxingGuard {
 public:
  template <typename... Tensors>
  explicit DeviceBoxingGuard(Tensors&&... tensors) {
    // Recording raw TensorImpl* is only safe if every boxed tensor outlives
    // the guard. Binding an rvalue (temporary) Tensor here would dangle after
    // the full expression, so reject temporaries at compile time -- callers
    // must pass named lvalue tensors.
    static_assert(
        (std::is_lvalue_reference_v<Tensors> && ...),
        "DeviceBoxingGuard must not box temporary (rvalue) tensors: "
        "the guard records raw TensorImpl* and does not extend lifetime");
    (maybe_box(tensors), ...);
  }
  ~DeviceBoxingGuard() {
    for (auto* impl : boxed_) {
      SetTensorImplDevice(impl, c10::DeviceType::PrivateUse1);
    }
  }
  DeviceBoxingGuard(const DeviceBoxingGuard&) = delete;
  DeviceBoxingGuard& operator=(const DeviceBoxingGuard&) = delete;
 private:
  void maybe_box(const at::Tensor& t) {
    if (t.defined() && t.is_privateuseone()) {
      auto* impl = t.unsafeGetTensorImpl();
      SetTensorImplDevice(impl, c10::DeviceType::CUDA);
      boxed_.push_back(impl);
    }
  }
  c10::SmallVector<c10::TensorImpl*, 4> boxed_;
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

// Materialize an ITensorListRef into a std::vector<at::Tensor>.
// The Tensor handles share the same TensorImpl as the originals, so boxing
// them (device metadata rewrite) affects the underlying tensors in place.
// The returned vector converts implicitly to at::TensorList (ArrayRef) for
// passing to PyTorch's public at:: API, which expects TensorList not IListRef.
inline std::vector<at::Tensor> MaterializeToTensorVec(
    const at::ITensorListRef& list) {
  std::vector<at::Tensor> out;
  out.reserve(list.size());
  for (const auto& t : list) {
    out.push_back(t);
  }
  return out;
}

// Drop "legacy empty" tensors (1-D with size 0) from a cat input list, matching
// ATen's native cat `should_skip` rule. maca's forked libtorch_cuda cat kernel
// takes a vectorized fast path when the non-empty tensor's numel is a multiple
// of 128 that does not honor this legacy skip, so it applies the cat dim against
// the empty tensor's 1-D rank and raises "Dimension out of range". Filtering
// here reproduces stock PyTorch semantics (e.g. transformers' KV-cache
// `torch.cat([torch.tensor([]), key_states], dim=-2)` on the first decode step).
// If every tensor is legacy-empty the list is returned unchanged so at::cat
// preserves its own empty-input behavior.
inline std::vector<at::Tensor> DropLegacyEmptyForCat(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> kept;
  kept.reserve(tensors.size());
  for (const auto& t : tensors) {
    if (t.defined() && t.dim() == 1 && t.sym_size(0) == 0) {
      continue;
    }
    kept.push_back(t);
  }
  if (kept.empty()) {
    return tensors;
  }
  return kept;
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
// leaving genuine CUDA tensors untouched. Boxed tensors are tracked by raw
// TensorImpl* (the list elements outlive the guard) to avoid refcount churn.
class TensorListBoxingGuard {
 public:
  TensorListBoxingGuard() = default;

  void box(at::TensorList tensors) {
    for (const auto& t : tensors) {
      if (t.defined() && t.is_privateuseone()) {
        auto* impl = t.unsafeGetTensorImpl();
        SetTensorImplDevice(impl, c10::DeviceType::CUDA);
        boxed_.push_back(impl);
      }
    }
  }

  // Track a tensor that was already boxed (for ITensorListRef iteration)
  void track(const at::Tensor& t) {
    if (t.defined()) boxed_.push_back(t.unsafeGetTensorImpl());
  }

  ~TensorListBoxingGuard() {
    for (auto* impl : boxed_) {
      SetTensorImplDevice(impl, c10::DeviceType::PrivateUse1);
    }
  }

  TensorListBoxingGuard(const TensorListBoxingGuard&) = delete;
  TensorListBoxingGuard& operator=(const TensorListBoxingGuard&) = delete;

 private:
  c10::SmallVector<c10::TensorImpl*, 4> boxed_;
};

} // namespace at::native::flagos
