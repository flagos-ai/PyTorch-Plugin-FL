// Copyright (c) 2026, BAAI. All rights reserved.
//
// Utilities for boxing/unboxing tensor device metadata between
// flagos (PrivateUse1) and CUDA. Since flagos and CUDA share the same
// GPU memory, temporarily changing device type in tensor metadata allows
// calling native PyTorch CUDA kernels that have device type assertions.

#pragma once

#include <ATen/autocast_mode.h>
#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/SmallVector.h>
#include <cstdlib>
#include <string>
#include <type_traits>

namespace at::native::flagos {

// Whether a boxing kernel may skip the inner PrivateUse1->native second dispatch
// and jump straight to the CUDA backend kernel via `::redispatch(DispatchKeySet(CUDA))`.
//
// A boxing kernel relabels its flagos (PrivateUse1) inputs to CUDA metadata, then
// calls `at::<op>(...)`. That inner call is a FULL PyTorch dispatch: it recomputes
// the DispatchKeySet, runs the RecordFunction TLS check, and — crucially — when
// grad/autocast is active it re-enters the AutogradCUDA (or Autocast) VariableType
// kernel that BUILDS THE BACKWARD GRAPH. torch_fl registers AutogradPrivateUse1 as a
// pure fallthrough (see register.cc), so the autograd graph is created ONLY by this
// inner AutogradCUDA layer, never at the PrivateUse1 level.
//
// `::redispatch(DispatchKeySet(CUDA))` carries the CUDA backend key alone and thus
// bypasses Autograd/Autocast entirely. That is exactly what we want for inference
// (no_grad decode is the hot, CPU-dispatch-bound path) — same backend kernel, same
// bit-for-bit result, minus the inner key recomputation and trampoline. But under an
// active GradMode it would SILENTLY DROP grad_fn recording and break training's
// backward pass. So gate the fast path on grad AND autocast both being disabled.
//
// Inputs are relabeled to CUDA before dispatch, so the relevant autocast state is the
// CUDA one (matching where the inner autograd/autocast keys would have fired).
//
// Beyond grad/autocast, the plain `at::<op>(...)` dispatch also honors the
// functionality keys that sit ABOVE the backend key: Functionalize, Python /
// PythonTLSSnapshot (torch_dispatch / __torch_function__ modes), and the
// Batched / FuncTorchBatched (vmap) wrappers. `::redispatch(DispatchKeySet(CUDA))`
// carries the CUDA backend key ALONE and silently skips all of them. Those keys
// arrive two ways: as a TLS *mode* on the current thread (no specific input tensor
// carries the bit) — checked here — or as a per-tensor lazy bit (Conjugate /
// Negative / ZeroTensor) riding on an individual input — checked by
// HasBoxingUnsafeKey() below, which the boxing guards fold into CanRedispatch().
// Either presence forces the safe `at::<op>(...)` path.
inline bool CanBoxingRedispatch() {
  // Kill switch for A/B measurement and emergency fallback: FLAGOS_NO_REDISPATCH
  // (any value other than "0") forces every boxing kernel back onto the plain
  // `at::<op>(...)` second dispatch. Matches the repo env-flag convention
  // (fallback.cc, caching_device_allocator.cc). Read once (the hot path calls
  // this per-op); default is the fast path enabled.
  static const bool disabled = [] {
    const char* e = std::getenv("FLAGOS_NO_REDISPATCH");
    return e != nullptr && std::string(e) != "0";
  }();
  if (disabled) return false;
  if (at::GradMode::is_enabled() ||
      at::autocast::is_autocast_enabled(c10::DeviceType::CUDA)) {
    return false;
  }
  // Ambient functionality MODES active on this thread (functionalize, vmap, a
  // Python/torch_dispatch mode). These are not attached to any single input, so
  // they must be caught here rather than per-tensor. included_ holds keys forced
  // on for the current TLS scope.
  constexpr c10::DispatchKeySet kUnsafeModes({
      c10::DispatchKey::Functionalize,
      c10::DispatchKey::Python,
      c10::DispatchKey::PythonTLSSnapshot,
      c10::DispatchKey::Batched,
      c10::DispatchKey::BatchedNestedTensor,
      c10::DispatchKey::FuncTorchBatched,
  });
  return !c10::impl::tls_local_dispatch_key_set().included_.has_any(kUnsafeModes);
}

// Per-tensor functionality bits that a bare CUDA redispatch would drop: the lazy
// Conjugate / Negative views and ZeroTensor, plus (defensively) a per-tensor
// Functionalize / Batched wrapper. If any boxed input carries one of these, the
// redispatch fast path is unsafe for that call and it must take `at::<op>(...)`,
// which materializes/handles the bit before reaching the backend kernel.
inline bool HasBoxingUnsafeKey(const at::Tensor& t) {
  if (!t.defined()) return false;
  constexpr c10::DispatchKeySet kUnsafeTensorKeys({
      c10::DispatchKey::Conjugate,
      c10::DispatchKey::Negative,
      c10::DispatchKey::ZeroTensor,
      c10::DispatchKey::Functionalize,
      c10::DispatchKey::Batched,
      c10::DispatchKey::BatchedNestedTensor,
      c10::DispatchKey::FuncTorchBatched,
  });
  return t.unsafeGetTensorImpl()->key_set().has_any(kUnsafeTensorKeys);
}

// --- SymInt argument adapters for the redispatch fast path ---------------------
// Generated wrappers use the NON-symint faithful signature (IntArrayRef / int64_t),
// matching PyTorch's PrivateUse1 registration. The public `at::<op>(...)` overload
// implicitly widens those to SymInt, but `at::_ops::<op>::redispatch(...)` uses the
// faithful SymInt signature verbatim and does NOT. So the redispatch branch widens
// symint args explicitly. Scalars/arrays use fromIntArrayRefSlow (a zero-copy
// reinterpret view — SymInt and int64_t are layout-compatible for the non-symbolic
// case) and c10::SymInt(x). The two optional forms need a materialized holder.
inline at::OptionalSymIntArrayRef ToOptSymIntArrayRef(
    at::OptionalIntArrayRef v) {
  if (!v.has_value()) return ::std::nullopt;
  return at::OptionalSymIntArrayRef(c10::fromIntArrayRefSlow(*v));
}

inline ::std::optional<c10::SymInt> ToOptSymInt(::std::optional<int64_t> v) {
  if (!v.has_value()) return ::std::nullopt;
  return ::std::optional<c10::SymInt>(c10::SymInt(*v));
}

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

  // Fast-path gate for this call: the global grad/autocast/mode check AND no
  // boxed input carrying a per-tensor conj/neg/zerotensor (etc.) bit. Generated
  // kernels call this instead of the bare CanBoxingRedispatch() so a lazy bit on
  // an actual input demotes just that call to the safe at::<op>(...) path.
  bool CanRedispatch() const {
    return !has_unsafe_key_ && CanBoxingRedispatch();
  }

 private:
  void BoxOne(const at::Tensor& t) {
    if (!t.defined()) return;
    // Check for redispatch-unsafe lazy bits on EVERY input (CPU/CUDA scalars can
    // carry conj/neg too), before device relabeling touches the keyset.
    has_unsafe_key_ = has_unsafe_key_ || HasBoxingUnsafeKey(t);
    if (t.is_privateuseone()) {
      auto* impl = t.unsafeGetTensorImpl();
      boxed_.push_back(impl);
      SetTensorImplDevice(impl, c10::DeviceType::CUDA);
    }
  }

  // Inline storage covers common operator arities while retaining unbounded
  // overflow capacity. Raw pointers avoid intrusive refcount traffic.
  c10::SmallVector<c10::TensorImpl*, 4> boxed_;
  bool has_unsafe_key_ = false;
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
      if (!t.defined()) continue;
      has_unsafe_key_ = has_unsafe_key_ || HasBoxingUnsafeKey(t);
      if (t.is_privateuseone()) {
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

  // See DeviceBoxingGuard::CanRedispatch.
  bool CanRedispatch() const {
    return !has_unsafe_key_ && CanBoxingRedispatch();
  }

 private:
  // Wrapper arguments and materialized Tensor vectors outlive the guard.
  c10::SmallVector<c10::TensorImpl*, 4> boxed_;
  bool has_unsafe_key_ = false;
};

} // namespace at::native::flagos
