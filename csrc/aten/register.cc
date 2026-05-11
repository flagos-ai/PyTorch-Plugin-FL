// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "factory_ops/empty.h"
#include "factory_ops/strided_ops.h"
#include "factory_ops/copy_ops.h"
#include "factory_ops/set_ops.h"
#include "factory_ops/contiguous_ops.h"
#include "factory_ops/fallback.h"
#include "functional_ops/mm.h"
#include "functional_ops/bmm.h"
#include "functional_ops/cat_stub.h"
#include "functional_ops/embedding_stub.h"
#include "functional_ops/add_stub.h"
#include "functional_ops/mul_stub.h"
#include "functional_ops/silu_stub.h"
#include "functional_ops/rsqrt_stub.h"
#include "functional_ops/mean_stub.h"
#include "functional_ops/cos_stub.h"
#include "functional_ops/sin_stub.h"
#include "functional_ops/neg_stub.h"
#include "functional_ops/pow_stub.h"
#include "functional_ops/all_stub.h"
#include "functional_ops/softmax_stub.h"
#include "functional_ops/bitwise_and_stub.h"
#include "functional_ops/le_stub.h"
#include "functional_ops/where_stub.h"
#include "functional_ops/index_stub.h"
#include "functional_ops/new_ones_stub.h"
#include "functional_ops/scalar_tensor_stub.h"

#include <ATen/native/CPUFallback.h>

#include <torch/library.h>

namespace at::flagos {

namespace {

at::Tensor WrapperEmptyMemoryFormat(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  return at::native::flagos::empty_memory_format(
      size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

at::Tensor WrapperEmptyStrided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  return at::native::flagos::empty_strided(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor WrapperAsStrided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
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
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  return at::native::flagos::_copy_from(self, dst, non_blocking);
}

at::Tensor WrapperCopyFromAndResize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  return at::native::flagos::_copy_from_and_resize(self, dst);
}

at::Scalar WrapperLocalScalarDense(const at::Tensor& self) {
  return at::native::flagos::_local_scalar_dense(self);
}

at::Tensor& WrapperSetSourceTensor(
    at::Tensor& self,
    const at::Tensor& source) {
  return at::native::flagos::set_source_Tensor_(self, source);
}

at::Tensor& WrapperSetSourceStorage(at::Tensor& self, at::Storage source) {
  return at::native::flagos::set_source_Storage_(self, source);
}

at::Tensor& WrapperSetSourceStorageOffset(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  return at::native::flagos::set_source_Storage_storage_offset_(
      result, storage, storage_offset, size, stride);
}

at::Tensor WrapperView(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::flagos::view(self, size);
}

at::Tensor WrapperContiguous(
    const at::Tensor& self,
    c10::MemoryFormat memory_format) {
  return at::native::flagos::contiguous(self, memory_format);
}

at::Tensor WrapperClone(
    const at::Tensor& self,
    std::optional<c10::MemoryFormat> memory_format) {
  return at::native::flagos::clone(self, memory_format);
}

at::Tensor WrapperToCopy(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<c10::MemoryFormat> memory_format) {
  return at::native::flagos::_to_copy(self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

void WrapperCpuFallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  at::native::flagos::cpu_fallback(op, stack);
}

void WrapperRecordStream(at::Tensor& self, at::Stream s) {
  // No-op for flagos tensors.
  // flagos uses cudaMalloc directly (not a caching allocator),
  // so there is no need to track stream usage for memory management.
}

at::Tensor WrapperMm(const at::Tensor& self, const at::Tensor& mat2) {
  auto out = at::empty({self.size(0), mat2.size(1)}, self.options());
  at::native::flagos::StructuredMmOutFlagos op(out);
  op.meta(self, mat2);
  op.impl(self, mat2, "mm");
  return out;
}

at::Tensor& WrapperMmOut(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::native::flagos::StructuredMmOutFlagos op(out);
  op.meta(self, mat2);
  op.impl(self, mat2, "mm.out");
  return out;
}

at::Tensor WrapperBmm(const at::Tensor& self, const at::Tensor& mat2) {
  auto out = at::empty({self.size(0), self.size(1), mat2.size(2)}, self.options());
  at::native::flagos::StructuredBmmOutFlagos op(out);
  op.meta(self, mat2);
  op.impl(self, mat2, "bmm");
  return out;
}

at::Tensor& WrapperBmmOut(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::native::flagos::StructuredBmmOutFlagos op(out);
  op.meta(self, mat2);
  op.impl(self, mat2, "bmm.out");
  return out;
}

at::Tensor WrapperCat(const at::ITensorListRef& tensors, int64_t dim) {
  return at::native::flagos::cat_stub(tensors, dim);
}

at::Tensor WrapperEmbedding(
    const at::Tensor& weight, const at::Tensor& indices,
    c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) {
  return at::native::flagos::embedding_stub(
      weight, indices, padding_idx.expect_int(), scale_grad_by_freq, sparse);
}

at::Tensor WrapperAddTensor(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  return at::native::flagos::add_tensor_stub(self, other, alpha);
}

at::Tensor WrapperMulTensor(
    const at::Tensor& self, const at::Tensor& other) {
  return at::native::flagos::mul_tensor_stub(self, other);
}

at::Tensor WrapperSilu(const at::Tensor& self) {
  return at::native::flagos::silu_stub(self);
}

at::Tensor WrapperRsqrt(const at::Tensor& self) {
  return at::native::flagos::rsqrt_stub(self);
}

at::Tensor WrapperMeanDim(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    bool keepdim, std::optional<at::ScalarType> dtype) {
  return at::native::flagos::mean_dim_stub(self, dim, keepdim, dtype);
}

at::Tensor WrapperCos(const at::Tensor& self) {
  return at::native::flagos::cos_stub(self);
}

at::Tensor WrapperSin(const at::Tensor& self) {
  return at::native::flagos::sin_stub(self);
}

at::Tensor WrapperNeg(const at::Tensor& self) {
  return at::native::flagos::neg_stub(self);
}

at::Tensor WrapperPowTensorScalar(const at::Tensor& self, const at::Scalar& exp) {
  return at::native::flagos::pow_tensor_scalar_stub(self, exp);
}

at::Tensor WrapperAll(const at::Tensor& self) {
  return at::native::flagos::all_stub(self);
}

at::Tensor WrapperSoftmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  return at::native::flagos::softmax_stub(self, dim, half_to_float);
}

at::Tensor WrapperBitwiseAndTensor(const at::Tensor& self, const at::Tensor& other) {
  return at::native::flagos::bitwise_and_tensor_stub(self, other);
}

at::Tensor WrapperLeTensor(const at::Tensor& self, const at::Tensor& other) {
  return at::native::flagos::le_tensor_stub(self, other);
}

at::Tensor WrapperWhereSelf(
    const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other) {
  return at::native::flagos::where_self_stub(condition, self, other);
}

at::Tensor WrapperIndexTensor(
    const at::Tensor& self, const c10::List<::std::optional<at::Tensor>>& indices) {
  return at::native::flagos::index_tensor_stub(self, indices);
}

at::Tensor WrapperNewOnes(
    const at::Tensor& self, at::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  return at::native::flagos::new_ones_stub(self, size, dtype, layout, device, pin_memory);
}

at::Tensor WrapperScalarTensor(
    const at::Scalar& s,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  return at::native::flagos::scalar_tensor_stub(s, dtype, layout, device, pin_memory);
}

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
  m.impl("_local_scalar_dense", WrapperLocalScalarDense);
  m.impl("set_.source_Tensor", WrapperSetSourceTensor);
  m.impl("set_.source_Storage", WrapperSetSourceStorage);
  m.impl(
      "set_.source_Storage_storage_offset",
      WrapperSetSourceStorageOffset);
  m.impl("view", WrapperView);
  m.impl("contiguous", WrapperContiguous);
  m.impl("clone", WrapperClone);
  m.impl("_to_copy", WrapperToCopy);
  m.impl("record_stream", WrapperRecordStream);
  m.impl("mm", WrapperMm);
  m.impl("mm.out", WrapperMmOut);
  m.impl("bmm", WrapperBmm);
  m.impl("bmm.out", WrapperBmmOut);
  m.impl("cat", WrapperCat);
  m.impl("embedding", WrapperEmbedding);
  m.impl("add.Tensor", WrapperAddTensor);
  m.impl("mul.Tensor", WrapperMulTensor);
  m.impl("silu", WrapperSilu);
  m.impl("rsqrt", WrapperRsqrt);
  m.impl("mean.dim", WrapperMeanDim);
  m.impl("cos", WrapperCos);
  m.impl("sin", WrapperSin);
  m.impl("neg", WrapperNeg);
  m.impl("pow.Tensor_Scalar", WrapperPowTensorScalar);
  m.impl("all", WrapperAll);
  m.impl("_softmax", WrapperSoftmax);
  m.impl("bitwise_and.Tensor", WrapperBitwiseAndTensor);
  m.impl("le.Tensor", WrapperLeTensor);
  m.impl("where.self", WrapperWhereSelf);
  m.impl("index.Tensor", WrapperIndexTensor);
  m.impl("new_ones", WrapperNewOnes);
  m.impl("scalar_tensor", WrapperScalarTensor);
}

// Register fallback for all unimplemented operators
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&WrapperCpuFallback>());
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
