// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "strided_ops.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/transpose_native.h>
#include <ATen/ops/permute_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/slice_native.h>
#include <ATen/ops/narrow_native.h>
#include <ATen/ops/squeeze_native.h>
#include <ATen/ops/unsqueeze_native.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/detach_native.h>

namespace at::native::flagos {

at::Tensor as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  auto int_size = C10_AS_INTARRAYREF_SLOW(size);
  auto int_stride = C10_AS_INTARRAYREF_SLOW(stride);
  std::optional<int64_t> int_offset = storage_offset.has_value()
      ? std::optional<int64_t>(storage_offset->expect_int())
      : std::nullopt;
  return at::native::as_strided_tensorimpl(self, int_size, int_stride, int_offset);
}

const at::Tensor& resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize_(
      self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
}

at::Tensor _reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(
      self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

at::Tensor expand(const at::Tensor& self, c10::SymIntArrayRef size, bool implicit) {
  return at::native::expand(self, C10_AS_INTARRAYREF_SLOW(size), implicit);
}

// NOTE: all view ops call at::native:: directly (not the tensor member method).
// The member methods re-dispatch through PrivateUse1, which routes back here and
// causes infinite recursion -> stack overflow. at::native:: are the raw stride
// implementations that operate on metadata without re-dispatching.
at::Tensor narrow(const at::Tensor& self, int64_t dim, int64_t start, int64_t length) {
  // at::native::narrow_symint computes the slice bounds and calls at::slice_symint,
  // which re-dispatches to the registered flagos slice_tensor (pure metadata, no
  // recursion). Calling self.narrow(...) here would re-enter this same kernel via
  // PrivateUse1 -> infinite recursion -> stack overflow (SIGSEGV).
  return at::native::narrow_symint(self, dim, start, length);
}

at::Tensor transpose_int(const at::Tensor& self, int64_t dim0, int64_t dim1) {
  return at::native::transpose(self, dim0, dim1);
}

at::Tensor permute(const at::Tensor& self, at::IntArrayRef dims) {
  return at::native::permute(self, dims);
}

at::Tensor select_int(const at::Tensor& self, int64_t dim, int64_t index) {
  return at::native::select_symint(self, dim, index);
}

at::Tensor slice_tensor(const at::Tensor& self, int64_t dim, ::std::optional<int64_t> start, ::std::optional<int64_t> end, int64_t step) {
  return at::native::slice(self, dim, start, end, step);
}

at::Tensor squeeze(const at::Tensor& self) {
  return at::native::squeeze(self);
}

at::Tensor squeeze_dim(const at::Tensor& self, int64_t dim) {
  return at::native::squeeze(self, dim);
}

at::Tensor unsqueeze(const at::Tensor& self, int64_t dim) {
  return at::native::unsqueeze(self, dim);
}

at::Tensor unsafe_view(const at::Tensor& self, at::IntArrayRef size) {
  return at::native::_unsafe_view(self, size);
}

// Pure-stride view: at::native::unfold computes the unfolded strides and calls
// as_strided, which re-dispatches to the registered flagos as_strided (metadata
// only, no recursion). Missing this registration made Tensor.repeat() fall back
// to an invalid CPU view op and return uninitialized data.
at::Tensor unfold(const at::Tensor& self, int64_t dimension, int64_t size, int64_t step) {
  return at::native::unfold(self, dimension, size, step);
}

at::Tensor detach(const at::Tensor& self) {
  return at::native::detach(self);
}

} // namespace at::native::flagos
