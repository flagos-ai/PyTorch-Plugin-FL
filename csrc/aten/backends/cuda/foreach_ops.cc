// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../foreach_ops.h"
#include "../../device_boxing.h"

#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_foreach_add.h>
#include <ATen/ops/_foreach_addcdiv.h>
#include <ATen/ops/_foreach_addcmul.h>
#include <ATen/ops/_foreach_lerp.h>
#include <ATen/ops/_foreach_sqrt.h>
#include <ATen/ops/_foreach_div.h>
#include <ATen/ops/_foreach_neg.h>
#include <ATen/ops/_foreach_reciprocal.h>

namespace at::native::flagos {

namespace {

// --- Inplace ops (return void) ---

void ForeachMulScalarKernelCuda(at::TensorList self, const at::Scalar& scalar) {
  TensorListBoxingGuard guard;
  guard.box(self);
  at::_foreach_mul_(self, scalar);
}

void ForeachAddScalarKernelCuda(at::TensorList self, const at::Scalar& scalar) {
  TensorListBoxingGuard guard;
  guard.box(self);
  at::_foreach_add_(self, scalar);
}

void ForeachAddcdivScalarListKernelCuda(
    at::TensorList self, at::TensorList tensor1, at::TensorList tensor2,
    at::ArrayRef<at::Scalar> scalars) {
  TensorListBoxingGuard guard;
  guard.box(self);
  guard.box(tensor1);
  guard.box(tensor2);
  at::_foreach_addcdiv_(self, tensor1, tensor2, scalars);
}

void ForeachAddcmulScalarKernelCuda(
    at::TensorList self, at::TensorList tensor1, at::TensorList tensor2,
    const at::Scalar& scalar) {
  TensorListBoxingGuard guard;
  guard.box(self);
  guard.box(tensor1);
  guard.box(tensor2);
  at::_foreach_addcmul_(self, tensor1, tensor2, scalar);
}

void ForeachLerpScalarKernelCuda(
    at::TensorList self, at::TensorList tensors1, const at::Scalar& weight) {
  TensorListBoxingGuard guard;
  guard.box(self);
  guard.box(tensors1);
  at::_foreach_lerp_(self, tensors1, weight);
}

void ForeachDivScalarListKernelCuda(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  TensorListBoxingGuard guard;
  guard.box(self);
  at::_foreach_div_(self, scalars);
}

void ForeachAddTensorListKernelCuda(
    at::TensorList self, at::TensorList other, const at::Scalar& alpha) {
  TensorListBoxingGuard guard;
  guard.box(self);
  guard.box(other);
  at::_foreach_add_(self, other, alpha);
}

void ForeachMulTensorListKernelCuda(
    at::TensorList self, at::TensorList other) {
  TensorListBoxingGuard guard;
  guard.box(self);
  guard.box(other);
  at::_foreach_mul_(self, other);
}

// --- Non-inplace ops (return vector<Tensor>) ---

::std::vector<at::Tensor> ForeachSqrtKernelCuda(at::TensorList self) {
  TensorListBoxingGuard guard;
  guard.box(self);
  auto result = at::_foreach_sqrt(self);
  UnboxTensorVecToFlagos(result);
  return result;
}

::std::vector<at::Tensor> ForeachNegKernelCuda(at::TensorList self) {
  TensorListBoxingGuard guard;
  guard.box(self);
  auto result = at::_foreach_neg(self);
  UnboxTensorVecToFlagos(result);
  return result;
}

::std::vector<at::Tensor> ForeachReciprocalKernelCuda(at::TensorList self) {
  TensorListBoxingGuard guard;
  guard.box(self);
  auto result = at::_foreach_reciprocal(self);
  UnboxTensorVecToFlagos(result);
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(ForeachMulScalarFn, foreach_mul_scalar_dispatcher, Backend::kCuda, ForeachMulScalarKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(ForeachAddScalarFn, foreach_add_scalar_dispatcher, Backend::kCuda, ForeachAddScalarKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(ForeachAddcdivScalarListFn, foreach_addcdiv_scalarlist_dispatcher, Backend::kCuda, ForeachAddcdivScalarListKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(ForeachAddcmulScalarFn, foreach_addcmul_scalar_dispatcher, Backend::kCuda, ForeachAddcmulScalarKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(ForeachLerpScalarFn, foreach_lerp_scalar_dispatcher, Backend::kCuda, ForeachLerpScalarKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(ForeachDivScalarListFn, foreach_div_scalarlist_dispatcher, Backend::kCuda, ForeachDivScalarListKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(ForeachAddTensorListFn, foreach_add_tensorlist_dispatcher, Backend::kCuda, ForeachAddTensorListKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(ForeachMulTensorListFn, foreach_mul_tensorlist_dispatcher, Backend::kCuda, ForeachMulTensorListKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(ForeachSqrtFn, foreach_sqrt_dispatcher, Backend::kCuda, ForeachSqrtKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(ForeachNegFn, foreach_neg_dispatcher, Backend::kCuda, ForeachNegKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(ForeachReciprocalFn, foreach_reciprocal_dispatcher, Backend::kCuda, ForeachReciprocalKernelCuda)

} // namespace at::native::flagos
