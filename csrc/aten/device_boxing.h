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

// Change a TensorImpl's device type in-place (metadata only, no data copy).
// Modifies dispatch key set, DataPtr device, and device_opt_.
inline void SetTensorImplDevice(
    c10::TensorImpl* impl, c10::DeviceType type) {
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
class DeviceBoxingGuard {
 public:
  // Pointer-only bookkeeping is valid for lvalue Tensor handles. Generated
  // wrapper arguments, optional holders, and local materialized vectors all
  // outlive the guard. Reject temporary Tensor handles at compile time.
  template <
      typename... Tensors,
      std::enable_if_t<
          (std::is_lvalue_reference_v<Tensors&&> && ...) &&
              (std::is_same_v<
                   std::remove_cv_t<std::remove_reference_t<Tensors>>,
                   at::Tensor> &&
               ...),
          int> = 0>
  explicit DeviceBoxingGuard(Tensors&&... tensors) {
    (BoxOne(tensors), ...);
  }

  ~DeviceBoxingGuard() {
    for (auto* impl : boxed_) {
      SetTensorImplDevice(impl, c10::DeviceType::PrivateUse1);
    }
  }

  DeviceBoxingGuard(const DeviceBoxingGuard&) = delete;
  DeviceBoxingGuard& operator=(const DeviceBoxingGuard&) = delete;

 private:
  void BoxOne(const at::Tensor& t) {
    if (t.defined() && t.is_privateuseone()) {
      auto* impl = t.unsafeGetTensorImpl();
      boxed_.push_back(impl);
      SetTensorImplDevice(impl, c10::DeviceType::CUDA);
    }
  }

  // Inline storage covers common operator arities while retaining unbounded
  // overflow capacity. Raw pointers avoid intrusive refcount traffic.
  c10::SmallVector<c10::TensorImpl*, 4> boxed_;
};

static_assert(
    std::is_constructible_v<DeviceBoxingGuard, at::Tensor&>,
    "DeviceBoxingGuard must accept mutable Tensor lvalues");
static_assert(
    std::is_constructible_v<DeviceBoxingGuard, const at::Tensor&>,
    "DeviceBoxingGuard must accept const Tensor lvalues");
static_assert(
    !std::is_constructible_v<DeviceBoxingGuard, at::Tensor&&>,
    "DeviceBoxingGuard must reject temporary Tensor handles");

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

// Materialize an ITensorListRef for general TensorList operators. The Tensor
// handles share their TensorImpl with the originals, so metadata boxing affects
// the original tensors and the vector can be passed as an at::TensorList.
inline std::vector<at::Tensor> MaterializeToTensorVec(
    const at::ITensorListRef& list) {
  std::vector<at::Tensor> out;
  out.reserve(list.size());
  for (const auto& t : list) {
    out.push_back(t);
  }
  return out;
}

// Materialize cat inputs while applying ATen's generic `should_skip` rule for
// legacy empty tensors (1-D with size 0). Inline capacity removes the common
// small-list heap allocation; SmallVector still grows for arbitrary list sizes.
// Preserve the original list when every input is skipped so at::cat retains its
// normal all-empty validation and result semantics.
//
// maca's forked libtorch_cuda cat kernel takes a vectorized fast path when the
// non-empty tensor's numel is a multiple of 128 that does not honor the legacy
// skip, so it applies the cat dim against the empty tensor's 1-D rank and raises
// "Dimension out of range". Filtering here reproduces stock PyTorch semantics
// (e.g. transformers' KV-cache `torch.cat([torch.tensor([]), key_states],
// dim=-2)` on the first decode step).
inline c10::SmallVector<at::Tensor, 4> MaterializeForCat(
    const at::ITensorListRef& list) {
  c10::SmallVector<at::Tensor, 4> kept;
  kept.reserve(list.size());
  for (const auto& t : list) {
    if (t.defined() && t.dim() == 1 && t.sym_size(0) == 0) {
      continue;
    }
    kept.push_back(t);
  }
  if (kept.empty()) {
    for (const auto& t : list) {
      kept.push_back(t);
    }
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
// leaving genuine CUDA tensors untouched.
class TensorListBoxingGuard {
 public:
  TensorListBoxingGuard() = default;

  void box(at::TensorList tensors) {
    for (const auto& t : tensors) {
      if (t.defined() && t.is_privateuseone()) {
        auto* impl = t.unsafeGetTensorImpl();
        boxed_.push_back(impl);
        SetTensorImplDevice(impl, c10::DeviceType::CUDA);
      }
    }
  }

  ~TensorListBoxingGuard() {
    for (auto* impl : boxed_) {
      SetTensorImplDevice(impl, c10::DeviceType::PrivateUse1);
    }
  }

  TensorListBoxingGuard(const TensorListBoxingGuard&) = delete;
  TensorListBoxingGuard& operator=(const TensorListBoxingGuard&) = delete;

 private:
  // Wrapper arguments and materialized Tensor vectors outlive the guard.
  c10::SmallVector<c10::TensorImpl*, 4> boxed_;
};

} // namespace at::native::flagos
