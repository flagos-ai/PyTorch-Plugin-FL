// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "strided_ops.h"
#include "generated/ops.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/transpose_native.h>
#include <ATen/ops/permute_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/slice_native.h>
#include <ATen/ops/squeeze_native.h>
#include <ATen/ops/unsqueeze_native.h>
#include <ATen/ops/_unsafe_view_native.h>
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

at::Tensor narrow(const at::Tensor& self, int64_t dim, int64_t start, int64_t length) {
  return self.narrow(dim, start, length);
}

// NOTE: all view ops call at::native:: directly (not the tensor member method).
// The member methods re-dispatch through PrivateUse1, which routes back here and
// causes infinite recursion -> stack overflow. at::native:: are the raw stride
// implementations that operate on metadata without re-dispatching.
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

at::Tensor detach(const at::Tensor& self) {
  return at::native::detach(self);
}

// View ops are pure metadata (stride) operations; they route through the
// generated dispatchers but need a backend kernel registered. Register them
// for the Ascend backend so the generated wrappers in register.inc resolve.
REGISTER_IMPL_TO_DISPATCHER(
    TransposeIntFn,
    transpose_int_dispatcher,
    Backend::kAscend,
    transpose_int)

REGISTER_IMPL_TO_DISPATCHER(
    PermuteFn,
    permute_dispatcher,
    Backend::kAscend,
    permute)

REGISTER_IMPL_TO_DISPATCHER(
    SelectIntFn,
    select_int_dispatcher,
    Backend::kAscend,
    select_int)

REGISTER_IMPL_TO_DISPATCHER(
    SliceTensorFn,
    slice_tensor_dispatcher,
    Backend::kAscend,
    slice_tensor)

REGISTER_IMPL_TO_DISPATCHER(
    SqueezeFn,
    squeeze_dispatcher,
    Backend::kAscend,
    squeeze)

REGISTER_IMPL_TO_DISPATCHER(
    SqueezeDimFn,
    squeeze_dim_dispatcher,
    Backend::kAscend,
    squeeze_dim)

REGISTER_IMPL_TO_DISPATCHER(
    UnsqueezeFn,
    unsqueeze_dispatcher,
    Backend::kAscend,
    unsqueeze)

REGISTER_IMPL_TO_DISPATCHER(
    PrivUnsafeViewFn,
    priv_unsafe_view_dispatcher,
    Backend::kAscend,
    unsafe_view)

REGISTER_IMPL_TO_DISPATCHER(
    DetachFn,
    detach_dispatcher,
    Backend::kAscend,
    detach)

} // namespace at::native::flagos
