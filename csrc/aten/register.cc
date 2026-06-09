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
#include "mm.h"
#include "add.h"
#include "silu.h"
#include "neg.h"
#include "bmm.h"
#include "cat.h"
#include "embedding.h"
#include "mul.h"
#include "rsqrt.h"
#include "mean.h"
#include "cos.h"
#include "sin.h"
#include "pow.h"
#include "all.h"
#include "softmax.h"
#include "bitwise_and.h"
#include "le.h"
#include "where.h"
#include "index.h"
#include "new_ones.h"
#include "scalar_tensor.h"
#include "ones_like.h"
#include "zeros.h"
#include "silu_backward.h"
#include "sum.h"
#include "slice_backward.h"
#include "constant_pad_nd.h"
#include "embedding_dense_backward.h"
#include "nll_loss.h"
#include "abs.h"
#include "acos.h"

#include <ATen/native/CPUFallback.h>
#include <ATen/ops/_index_put_impl.h>
#include <ATen/ops/index_put.h>

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

at::Tensor& WrapperCopy_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  at::native::flagos::_copy_from(src, self, non_blocking);
  return self;
}

at::Scalar WrapperLocalScalarDense(const at::Tensor& self) {
  return at::native::flagos::local_scalar_dense_dispatcher(self);
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
  return at::native::flagos::to_copy_dispatcher(
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
  at::native::flagos::StructuredMmOut op(out);
  op.meta(self, mat2);
  op.impl(self, mat2, "mm");
  return out;
}

at::Tensor& WrapperMmOut(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::native::flagos::StructuredMmOut op(out);
  op.meta(self, mat2);
  op.impl(self, mat2, "mm.out");
  return out;
}



at::Tensor WrapperAddTensor(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  return at::native::flagos::add_tensor_dispatcher(self, other, alpha);
}

at::Tensor WrapperAddScalar(
    const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
  auto other_tensor = at::scalar_tensor(other, self.options());
  auto alpha_val = alpha.toDouble();
  if (alpha_val != 1.0) {
    other_tensor = other_tensor * alpha;
  }
  return at::native::flagos::add_tensor_dispatcher(self, other_tensor, at::Scalar(1));
}

at::Tensor WrapperSilu(const at::Tensor& self) {
  return at::native::flagos::silu_dispatcher(self);
}

at::Tensor WrapperNeg(const at::Tensor& self) {
  return at::native::flagos::neg_dispatcher(self);
}

at::Tensor WrapperBmm(const at::Tensor& self, const at::Tensor& mat2) {
  auto out = at::empty({self.size(0), self.size(1), mat2.size(2)}, self.options());
  at::native::flagos::StructuredBmmOut op(out);
  op.meta(self, mat2);
  op.impl(self, mat2, "bmm");
  return out;
}

at::Tensor& WrapperBmmOut(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::native::flagos::StructuredBmmOut op(out);
  op.meta(self, mat2);
  op.impl(self, mat2, "bmm.out");
  return out;
}

at::Tensor WrapperCat(const at::ITensorListRef& tensors, int64_t dim) {
  return at::native::flagos::cat_dispatcher(tensors, dim);
}

at::Tensor WrapperEmbedding(
    const at::Tensor& weight, const at::Tensor& indices,
    c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) {
  return at::native::flagos::embedding_dispatcher(
      weight, indices, padding_idx.expect_int(), scale_grad_by_freq, sparse);
}

at::Tensor WrapperMulTensor(
    const at::Tensor& self, const at::Tensor& other) {
  return at::native::flagos::mul_tensor_dispatcher(self, other);
}

at::Tensor WrapperRsqrt(const at::Tensor& self) {
  return at::native::flagos::rsqrt_dispatcher(self);
}

at::Tensor WrapperMeanDim(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    bool keepdim, std::optional<at::ScalarType> dtype) {
  return at::native::flagos::mean_dim_dispatcher(self, dim, keepdim, dtype);
}

at::Tensor WrapperCos(const at::Tensor& self) {
  return at::native::flagos::cos_dispatcher(self);
}

at::Tensor WrapperSin(const at::Tensor& self) {
  return at::native::flagos::sin_dispatcher(self);
}

at::Tensor WrapperPowTensorScalar(const at::Tensor& self, const at::Scalar& exp) {
  return at::native::flagos::pow_tensor_scalar_dispatcher(self, exp);
}

at::Tensor WrapperAll(const at::Tensor& self) {
  return at::native::flagos::all_dispatcher(self);
}

at::Tensor WrapperSoftmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  return at::native::flagos::softmax_dispatcher(self, dim, half_to_float);
}

at::Tensor WrapperBitwiseAndTensor(const at::Tensor& self, const at::Tensor& other) {
  return at::native::flagos::bitwise_and_tensor_dispatcher(self, other);
}

at::Tensor WrapperLeTensor(const at::Tensor& self, const at::Tensor& other) {
  return at::native::flagos::le_tensor_dispatcher(self, other);
}

at::Tensor WrapperWhereSelf(
    const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other) {
  return at::native::flagos::where_self_dispatcher(condition, self, other);
}

at::Tensor WrapperIndexTensor(
    const at::Tensor& self, const c10::List<::std::optional<at::Tensor>>& indices) {
  return at::native::flagos::index_tensor_dispatcher(self, indices);
}

at::Tensor WrapperNewOnes(
    const at::Tensor& self, at::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  return at::native::flagos::new_ones_dispatcher(self, size, dtype, layout, device, pin_memory);
}

at::Tensor WrapperScalarTensor(
    const at::Scalar& s,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  return at::native::flagos::scalar_tensor_dispatcher(s, dtype, layout, device, pin_memory);
}

at::Tensor WrapperOnesLike(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    std::optional<at::MemoryFormat> memory_format) {
  return at::native::flagos::ones_like_dispatcher(self, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor WrapperZeros(
    at::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  return at::native::flagos::zeros_dispatcher(size, dtype, layout, device, pin_memory);
}

at::Tensor WrapperSiluBackward(const at::Tensor& grad_output, const at::Tensor& self) {
  return at::native::flagos::silu_backward_dispatcher(grad_output, self);
}

at::Tensor WrapperSumDimIntList(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    bool keepdim, std::optional<at::ScalarType> dtype) {
  return at::native::flagos::sum_dim_dispatcher(self, dim, keepdim, dtype);
}

at::Tensor WrapperSliceBackward(
    const at::Tensor& grad_output, at::IntArrayRef input_sizes,
    int64_t dim, int64_t start, int64_t end, int64_t step) {
  return at::native::flagos::slice_backward_dispatcher(grad_output, input_sizes, dim, start, end, step);
}

at::Tensor WrapperConstantPadNd(
    const at::Tensor& self, at::IntArrayRef pad, const at::Scalar& value) {
  return at::native::flagos::constant_pad_nd_dispatcher(self, pad, value);
}

at::Tensor WrapperEmbeddingDenseBackward(
    const at::Tensor& grad_output, const at::Tensor& indices,
    int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  return at::native::flagos::embedding_dense_backward_dispatcher(
      grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
}

std::tuple<at::Tensor, at::Tensor> WrapperNllLossForward(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index) {
  return at::native::flagos::nll_loss_forward_dispatcher(self, target, weight, reduction, ignore_index);
}

at::Tensor WrapperNllLossBackward(
    const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index, const at::Tensor& total_weight) {
  return at::native::flagos::nll_loss_backward_dispatcher(
      grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

at::Tensor WrapperAbs(const at::Tensor& self) {
  return at::native::flagos::abs_dispatcher(self);
}

at::Tensor WrapperAcos(const at::Tensor& self) {
  return at::native::flagos::acos_dispatcher(self);
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
  m.impl("copy_", WrapperCopy_);
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
  m.impl("index_put_", WrapperIndexPut_);
  m.impl("_index_put_impl_", WrapperIndexPutImpl_);
  m.impl("record_stream", WrapperRecordStream);
  m.impl("mm", WrapperMm);
  m.impl("mm.out", WrapperMmOut);
  m.impl("add.Tensor", WrapperAddTensor);
  m.impl("add.Scalar", WrapperAddScalar);
  m.impl("silu", WrapperSilu);
  m.impl("neg", WrapperNeg);
  m.impl("bmm", WrapperBmm);
  m.impl("bmm.out", WrapperBmmOut);
  m.impl("cat", WrapperCat);
  m.impl("embedding", WrapperEmbedding);
  m.impl("mul.Tensor", WrapperMulTensor);
  m.impl("rsqrt", WrapperRsqrt);
  m.impl("mean.dim", WrapperMeanDim);
  m.impl("cos", WrapperCos);
  m.impl("sin", WrapperSin);
  m.impl("pow.Tensor_Scalar", WrapperPowTensorScalar);
  m.impl("all", WrapperAll);
  m.impl("_softmax", WrapperSoftmax);
  m.impl("bitwise_and.Tensor", WrapperBitwiseAndTensor);
  m.impl("le.Tensor", WrapperLeTensor);
  m.impl("where.self", WrapperWhereSelf);
  m.impl("index.Tensor", WrapperIndexTensor);
  m.impl("new_ones", WrapperNewOnes);
  m.impl("scalar_tensor", WrapperScalarTensor);
  m.impl("ones_like", WrapperOnesLike);
  m.impl("zeros", WrapperZeros);
  m.impl("silu_backward", WrapperSiluBackward);
  m.impl("sum.dim_IntList", WrapperSumDimIntList);
  m.impl("slice_backward", WrapperSliceBackward);
  m.impl("constant_pad_nd", WrapperConstantPadNd);
  m.impl("embedding_dense_backward", WrapperEmbeddingDenseBackward);
  m.impl("nll_loss_forward", WrapperNllLossForward);
  m.impl("nll_loss_backward", WrapperNllLossBackward);
  m.impl("abs", WrapperAbs);
  m.impl("acos", WrapperAcos);
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
