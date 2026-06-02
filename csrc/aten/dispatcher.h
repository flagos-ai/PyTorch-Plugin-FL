// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "common.h"
#include <c10/util/Exception.h>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace at::native::flagos {

// Lightweight op dispatcher replacing PyTorch's DispatchStub.
//
// A dispatcher owns an op_name (e.g. "mm") and a set of per-backend kernel
// pointers. Multiple ops can share one dispatcher (e.g. "mm" and "mm.out"
// share the same kernels). The op name used for config lookup and
// logging is passed at the call site via DispatchAs(), or defaults
// to the dispatcher's op_name via operator().
//
// Usage:
//   // header:
//   using MmFn = void(*)(const Tensor&, const Tensor&, Tensor&);
//   DECLARE_DISPATCHER(MmFn, mm_dispatcher)
//
//   // cpp:
//   ADD_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, "mm")
//   REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kFlagOs, my_flagos_mm)
//   REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kCuda,   my_cuda_mm)
//
//   // call (uses op_name "mm" for config lookup):
//   mm_dispatcher(self, mat2, out);
//
//   // call with op name override (uses "mm.out" for config lookup):
//   mm_dispatcher.DispatchAs("mm.out", self, mat2, out);

template <typename FnPtr>
class Dispatcher {
 public:
  // constexpr constructor ensures constant initialization (placed in .bss/.data
  // at load time), which is guaranteed to complete before any dynamic
  // initialization (DispatchRegistrar constructors). This eliminates the
  // static initialization order fiasco when DEFINE and REGISTER are in
  // different translation units.
  constexpr explicit Dispatcher(const char* op_name)
      : op_name_(op_name) {}

  void RegisterKernel(Backend device, FnPtr fn) {
    switch (device) {
      case Backend::kCuda:          cuda_fn_ = fn;           break;
      case Backend::kFlagOs:        flagos_fn_ = fn;         break;
      case Backend::kFlagOsPython:  flagos_python_fn_ = fn;  break;
      case Backend::kAscend:        ascend_fn_ = fn;         break;
      case Backend::kMusa:          musa_fn_ = fn;           break;
      case Backend::kMetax:         metax_fn_ = fn;          break;
    }
  }

  template <typename... Args>
  auto operator()(Args&&... args) const {
    return DispatchAs(op_name_, std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto DispatchAs(const std::string& op_name, Args&&... args) const {
    auto backend = GetBackendForOp(op_name);
    LogDispatch(op_name, backend);
    auto fn = GetFn(backend);
    TORCH_CHECK(fn, op_name, ": backend not registered");
    return fn(std::forward<Args>(args)...);
  }

 private:
  FnPtr GetFn(Backend device) const {
    switch (device) {
      case Backend::kCuda:          return cuda_fn_;
      case Backend::kFlagOs:        return flagos_fn_;
      case Backend::kFlagOsPython:  return flagos_python_fn_;
      case Backend::kAscend:        return ascend_fn_;
      case Backend::kMusa:          return musa_fn_;
      case Backend::kMetax:         return metax_fn_;
    }
    return nullptr;
  }

  static void LogDispatch(const std::string& op_name, Backend backend) {
    static const bool enabled = []() {
      const char* v = std::getenv("FLAGOS_LOG_DISPATCH");
      return v && std::string(v) == "1";
    }();
    if (!enabled) return;
    const char* name;
    switch (backend) {
      case Backend::kCuda:          name = "cuda"; break;
      case Backend::kFlagOs:        name = "flagos"; break;
      case Backend::kFlagOsPython:  name = "flagos_python"; break;
      case Backend::kAscend:        name = "ascend"; break;
      case Backend::kMusa:          name = "musa"; break;
      case Backend::kMetax:         name = "metax"; break;
      default:                           name = "unknown"; break;
    }
    fprintf(stderr, "[flagos dispatch] %s -> %s\n", op_name.c_str(), name);
  }

  const char* op_name_ = nullptr;
  FnPtr cuda_fn_           = nullptr;
  FnPtr flagos_fn_         = nullptr;
  FnPtr flagos_python_fn_  = nullptr;
  FnPtr ascend_fn_         = nullptr;
  FnPtr musa_fn_           = nullptr;
  FnPtr metax_fn_          = nullptr;
};

namespace detail {
template <typename FnPtr>
struct DispatchRegistrar {
  DispatchRegistrar(Dispatcher<FnPtr>& dispatcher, Backend device, FnPtr fn) {
    dispatcher.RegisterKernel(device, fn);
  }
};
} // namespace detail

} // namespace at::native::flagos

#define DECLARE_DISPATCHER(fn_type, name) \
  extern ::at::native::flagos::Dispatcher<fn_type> name;

#define ADD_IMPL_TO_DISPATCHER(fn_type, name, op_name) \
  ::at::native::flagos::Dispatcher<fn_type> name(op_name);

#define REGISTER_IMPL_TO_DISPATCHER_UID2(fn_type, name, device, fn, uid) \
  __attribute__((used)) static ::at::native::flagos::detail::DispatchRegistrar<fn_type>    \
      name##_registrar_##uid(name, device, fn);
#define REGISTER_IMPL_TO_DISPATCHER_UID(fn_type, name, device, fn, uid) \
  REGISTER_IMPL_TO_DISPATCHER_UID2(fn_type, name, device, fn, uid)
#define REGISTER_IMPL_TO_DISPATCHER(fn_type, name, device, fn) \
  REGISTER_IMPL_TO_DISPATCHER_UID(fn_type, name, device, fn, __COUNTER__)

// Transitional compatibility: keep old backend files buildable while moving
// from FLAGOS_* dispatch registration to Dispatcher + REGISTER_IMPL_*.
using FlagosDevice = ::at::native::flagos::Backend;
#define FLAGOS_REGISTER_DISPATCH(fn_type, stub_name, device, fn) \
  REGISTER_IMPL_TO_DISPATCHER(fn_type, stub_name, device, fn)

// Legacy stub-name aliases -> new dispatcher symbols.
#define zeros_stub zeros_dispatcher
#define where_self_stub where_self_dispatcher
#define sum_dim_stub sum_dim_dispatcher
#define softmax_stub softmax_dispatcher
#define slice_backward_stub slice_backward_dispatcher
#define sin_stub sin_dispatcher
#define silu_backward_stub silu_backward_dispatcher
#define silu_stub silu_dispatcher
#define scalar_tensor_stub scalar_tensor_dispatcher
#define rsqrt_stub rsqrt_dispatcher
#define pow_tensor_scalar_stub pow_tensor_scalar_dispatcher
#define ones_like_stub ones_like_dispatcher
#define nll_loss_forward_stub nll_loss_forward_dispatcher
#define nll_loss_backward_stub nll_loss_backward_dispatcher
#define new_ones_stub new_ones_dispatcher
#define neg_stub neg_dispatcher
#define mm_stub mm_dispatcher
#define mean_dim_stub mean_dim_dispatcher
#define index_tensor_stub index_tensor_dispatcher
#define embedding_dense_backward_stub embedding_dense_backward_dispatcher
#define embedding_stub embedding_dispatcher
#define cos_stub cos_dispatcher
#define constant_pad_nd_stub constant_pad_nd_dispatcher
#define cat_stub cat_dispatcher
#define bmm_stub bmm_dispatcher
#define bitwise_and_tensor_stub bitwise_and_tensor_dispatcher
#define mul_tensor_stub mul_tensor_dispatcher
#define add_tensor_stub add_tensor_dispatcher
#define le_tensor_stub le_tensor_dispatcher
#define all_stub all_dispatcher
