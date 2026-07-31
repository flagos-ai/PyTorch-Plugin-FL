// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "empty.h"
#include "strided_ops.h"
#include "copy_ops.h"
#include "copy_dispatcher.h"
#include "set_ops.h"
#include "contiguous_ops.h"
#include "fallback.h"

// Generated dispatcher headers
#include "generated/ops.h"

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/SDPBackend.h>
#include <torch/library.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include "common.h"
#include "runtime/allocator/caching_device_allocator.h"

namespace at::flagos {

namespace {

at::Tensor WrapperEmptyMemoryFormat(
    c10::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype_opt,
    ::std::optional<at::Layout> layout_opt,
    ::std::optional<at::Device> device_opt,
    ::std::optional<bool> pin_memory_opt,
    ::std::optional<at::MemoryFormat> memory_format_opt) {
  return at::native::flagos::empty_memory_format(
      size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

at::Tensor WrapperEmptyStrided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    ::std::optional<at::ScalarType> dtype_opt,
    ::std::optional<at::Layout> layout_opt,
    ::std::optional<at::Device> device_opt,
    ::std::optional<bool> pin_memory_opt) {
  return at::native::flagos::empty_strided(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor WrapperAsStrided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    ::std::optional<c10::SymInt> storage_offset) {
  return at::native::flagos::as_strided(self, size, stride, storage_offset);
}

const at::Tensor& WrapperResize(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::flagos::resize_(self, size, memory_format);
}

at::Tensor WrapperReshapeAlias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return at::native::flagos::_reshape_alias(self, size, stride);
}

at::Tensor WrapperCopyFrom(
    const at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  return at::native::flagos::_copy_from(self, src, non_blocking);
}

at::Tensor WrapperCopyFromAndResize(
    const at::Tensor& self, const at::Tensor& dst) {
  return at::native::flagos::_copy_from_and_resize(self, dst);
}

at::Tensor& WrapperCopy_(
    at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  at::native::flagos::_copy_from(src, self, non_blocking);
  return self;
}

at::Scalar WrapperLocalScalarDense(const at::Tensor& self) {
  return at::native::flagos::_local_scalar_dense(self);
}

at::Tensor& WrapperSetSourceTensor(
    at::Tensor& self, const at::Tensor& source) {
  return at::native::flagos::set_source_Tensor_(self, source);
}

at::Tensor& WrapperSetSourceStorage(at::Tensor& self, at::Storage source) {
  return at::native::flagos::set_source_Storage_(self, source);
}

at::Tensor& WrapperSetSourceStorageOffset(
    at::Tensor& self,
    at::Storage source,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  return at::native::flagos::set_source_Storage_storage_offset_(
      self, source, storage_offset, size, stride);
}

at::Tensor WrapperView(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::flagos::view(self, size);
}

at::Tensor WrapperExpand(const at::Tensor& self, c10::SymIntArrayRef size, bool implicit) {
  return at::native::flagos::expand(self, size, implicit);
}

at::Tensor WrapperNarrow(const at::Tensor& self, int64_t dim, int64_t start, int64_t length) {
  return at::native::flagos::narrow(self, dim, start, length);
}

// These operators only derive a new Tensor view from existing storage. Keeping
// them in the hand-written metadata path avoids device boxing and backend
// selection while preserving the normal ATen view and aliasing semantics.
at::Tensor WrapperTransposeInt(
    const at::Tensor& self, int64_t dim0, int64_t dim1) {
  return at::native::flagos::transpose_int(self, dim0, dim1);
}

at::Tensor WrapperPermute(const at::Tensor& self, at::IntArrayRef dims) {
  return at::native::flagos::permute(self, dims);
}

at::Tensor WrapperSelectInt(
    const at::Tensor& self, int64_t dim, int64_t index) {
  return at::native::flagos::select_int(self, dim, index);
}

at::Tensor WrapperSliceTensor(
    const at::Tensor& self,
    int64_t dim,
    ::std::optional<int64_t> start,
    ::std::optional<int64_t> end,
    int64_t step) {
  return at::native::flagos::slice_tensor(self, dim, start, end, step);
}

at::Tensor WrapperSqueeze(const at::Tensor& self) {
  return at::native::flagos::squeeze(self);
}

at::Tensor WrapperSqueezeDim(const at::Tensor& self, int64_t dim) {
  return at::native::flagos::squeeze_dim(self, dim);
}

at::Tensor WrapperUnsqueeze(const at::Tensor& self, int64_t dim) {
  return at::native::flagos::unsqueeze(self, dim);
}

at::Tensor WrapperUnsafeView(const at::Tensor& self, at::IntArrayRef size) {
  return at::native::flagos::unsafe_view(self, size);
}

at::Tensor WrapperDetach(const at::Tensor& self) {
  return at::native::flagos::detach(self);
}

// unfold is a pure-stride view op. Without a flagos registration it falls back
// to CPU, which is invalid for view ops (storage cannot be shared across
// devices) -> PyTorch emits a warning and returns an uninitialized tensor.
// This silently corrupts any op that relies on unfold internally, e.g.
// Tensor.repeat() (which unfolds the result buffer and copy_'s into it),
// producing garbage data. See GPTJ repeat/gather crash root-cause report.
at::Tensor WrapperUnfold(
    const at::Tensor& self, int64_t dimension, int64_t size, int64_t step) {
  return at::native::flagos::unfold(self, dimension, size, step);
}

at::Tensor WrapperContiguous(
    const at::Tensor& self, at::MemoryFormat memory_format) {
  return at::native::flagos::contiguous(self, memory_format);
}

at::Tensor WrapperClone(
    const at::Tensor& self,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::flagos::clone(self, memory_format);
}

at::Tensor WrapperToCopy(
    const at::Tensor& self,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    bool non_blocking,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::flagos::_to_copy(
      self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

at::Tensor& WrapperIndexPut_(
    at::Tensor& self,
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate) {
  at::Tensor self_cpu = self.cpu();
  at::Tensor values_cpu = values.cpu();
  c10::List<::std::optional<at::Tensor>> indices_cpu;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    auto opt = indices.get(i);
    if (opt.has_value() && opt->defined()) {
      indices_cpu.push_back(opt->cpu());
    } else {
      indices_cpu.push_back(std::nullopt);
    }
  }
  at::index_put_(self_cpu, indices_cpu, values_cpu, accumulate);
  self.copy_(self_cpu);
  return self;
}

at::Tensor& WrapperIndexPutImpl_(
    at::Tensor& self,
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate,
    bool /*unsafe*/) {
  return WrapperIndexPut_(self, indices, values, accumulate);
}

void WrapperRecordStream(at::Tensor& self, at::Stream s) {
  if (!c10::flagos::CachingDeviceAllocator::is_enabled()) {
    // No-op when caching allocator is disabled.
    return;
  }
  auto* alloc = c10::flagos::GetCachingAllocator();
  if (!alloc) {
    // No-op when caching allocator is not available on this platform.
    return;
  }
  // Convert at::Stream to flagos Stream_t.
  // The stream id encodes the underlying device stream pointer.
  Stream_t stream = reinterpret_cast<Stream_t>(s.id());
  alloc->record_stream(self.storage().data_ptr(), stream);
}

// _fused_sdp_choice: tells PyTorch's scaled_dot_product_attention which fused
// backend to use for PrivateUse1 tensors. Returning efficient_attention (2)
// routes to _scaled_dot_product_efficient_attention, which has an Ascend
// aclnnFlashAttentionScore kernel. Without this, SDPA falls back to the math
// decomposition path (_safe_softmax etc.), which is slower and needs many more
// ops registered.
int64_t WrapperFusedSdpChoice(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  return static_cast<int64_t>(at::SDPBackend::efficient_attention);
}

// ============================================================
// Generated wrappers for 71 CUDA operators
// ============================================================
#define FLAGOS_GEN_WRAPPERS
#include "generated/register.inc"
#undef FLAGOS_GEN_WRAPPERS

} // namespace

// Register basic operators for PrivateUse1 dispatch key
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", WrapperEmptyMemoryFormat);
  m.impl("empty_strided", WrapperEmptyStrided);
  m.impl("as_strided", WrapperAsStrided);
  m.impl("resize_", WrapperResize);
  m.impl("_reshape_alias", WrapperReshapeAlias);
  m.impl("_copy_from", WrapperCopyFrom);
  m.impl("_copy_from_and_resize", WrapperCopyFromAndResize);
  m.impl("copy_", WrapperCopy_);
  m.impl("_local_scalar_dense", WrapperLocalScalarDense);
  m.impl("set_.source_Tensor", WrapperSetSourceTensor);
  m.impl("set_.source_Storage", WrapperSetSourceStorage);
  m.impl(
      "set_.source_Storage_storage_offset", WrapperSetSourceStorageOffset);
  m.impl("view", WrapperView);
  m.impl("expand", WrapperExpand);
  m.impl("narrow", WrapperNarrow);
  m.impl("transpose.int", WrapperTransposeInt);
  m.impl("permute", WrapperPermute);
  m.impl("select.int", WrapperSelectInt);
  m.impl("slice.Tensor", WrapperSliceTensor);
  m.impl("squeeze", WrapperSqueeze);
  m.impl("squeeze.dim", WrapperSqueezeDim);
  m.impl("unsqueeze", WrapperUnsqueeze);
  m.impl("_unsafe_view", WrapperUnsafeView);
  m.impl("detach", WrapperDetach);
  m.impl("unfold", WrapperUnfold);
  m.impl("contiguous", WrapperContiguous);
  m.impl("clone", WrapperClone);
  m.impl("_to_copy", WrapperToCopy);
  m.impl("index_put_", WrapperIndexPut_);
  m.impl("_index_put_impl_", WrapperIndexPutImpl_);
  m.impl("record_stream", WrapperRecordStream);
  m.impl("_fused_sdp_choice", WrapperFusedSdpChoice);

  // ============================================================
  // Generated m.impl registrations for the generated operators
  // ============================================================
  // GCU registers only its own coverage set. Claiming an op on PrivateUse1
  // without a kernel behind it turns into the dispatcher's "backend not
  // registered" error, whereas leaving it unregistered reaches the cpu_fallback
  // below -- and on GCU neither the CUDA boxing kernels (no CUDA runtime) nor
  // FlagGems are built, so the full list would break every uncovered op.
  //
  // MUSA is the same story: musa_register.inc lists exactly the ops with a mudnn
  // kernel behind them, and no CUDA boxing kernels are compiled in
  // (musa_runtime, not cudart), so uncovered ops must fall through rather than
  // be claimed. The two convolution `*_overrideable` ops are the exception that
  // proves the rule -- ATen's default for them raises instead of being boxable,
  // so they get real kernels in backends/musa/mudnn_conv.cc.
  #if defined(USE_GCU)
    #if defined(FLAGOS_GCU_KERNEL)
    #include "backends/gcu/generated/gcu_register.inc"
    #endif
  #elif defined(USE_MUSA)
    #if defined(FLAGOS_MUSA_KERNEL)
    #include "backends/musa/generated/musa_register.inc"
    #endif
  #else
  #define FLAGOS_GEN_IMPLS
  #include "generated/register.inc"
  #undef FLAGOS_GEN_IMPLS
  #endif
}

// Register fallback for all unimplemented operators
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&at::native::flagos::cpu_fallback>());
}

// Register AutogradPrivateUse1 fallback to dispatch to PrivateUse1
// This ensures operators like where.ScalarSelf work correctly through autograd dispatch
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

// Register autograd-aware contiguous for PrivateUse1 tensors.
//
// Problem: contiguous registered on PrivateUse1 bypasses autograd recording
// (AutogradPrivateUse1 is fallthrough), causing grad_fn=None on the output
// and breaking gradient propagation (e.g., in attention layers that use
// transpose().contiguous()). On CUDA, contiguous() returns a tensor with
// CloneBackward0 grad_fn; on flagos it returned grad_fn=None.
//
// Solution: Register contiguous on AutogradPrivateUse1 so it intercepts
// the call before fallthrough. When the tensor actually needs copying
// (is non-contiguous), we use clone(memory_format) which properly records
// autograd operations. clone dispatches to PrivateUse1::clone which
// handles the actual data copy.
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl("contiguous", [](const at::Tensor& self, c10::MemoryFormat memory_format) -> at::Tensor {
    if (self.is_contiguous(memory_format)) {
      return self;
    }
    // clone(memory_format) creates a contiguous copy with autograd tracking.
    // This dispatches to PrivateUse1::clone (which uses empty + copy_),
    // and autograd records CloneBackward0 for gradient propagation.
    return self.clone(memory_format);
  });
}

} // namespace at::flagos
