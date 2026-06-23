// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include "dispatcher.h"

#include <vector>

namespace at::native::flagos {

// --- Inplace ops (return void) ---

using ForeachMulScalarFn = void (*)(at::TensorList, const at::Scalar&);
DECLARE_DISPATCHER(ForeachMulScalarFn, foreach_mul_scalar_dispatcher)

using ForeachAddScalarFn = void (*)(at::TensorList, const at::Scalar&);
DECLARE_DISPATCHER(ForeachAddScalarFn, foreach_add_scalar_dispatcher)

using ForeachAddcdivScalarListFn = void (*)(
    at::TensorList, at::TensorList, at::TensorList, at::ArrayRef<at::Scalar>);
DECLARE_DISPATCHER(ForeachAddcdivScalarListFn, foreach_addcdiv_scalarlist_dispatcher)

using ForeachAddcmulScalarFn = void (*)(
    at::TensorList, at::TensorList, at::TensorList, const at::Scalar&);
DECLARE_DISPATCHER(ForeachAddcmulScalarFn, foreach_addcmul_scalar_dispatcher)

using ForeachLerpScalarFn = void (*)(at::TensorList, at::TensorList, const at::Scalar&);
DECLARE_DISPATCHER(ForeachLerpScalarFn, foreach_lerp_scalar_dispatcher)

using ForeachDivScalarListFn = void (*)(at::TensorList, at::ArrayRef<at::Scalar>);
DECLARE_DISPATCHER(ForeachDivScalarListFn, foreach_div_scalarlist_dispatcher)

using ForeachAddTensorListFn = void (*)(at::TensorList, at::TensorList, const at::Scalar&);
DECLARE_DISPATCHER(ForeachAddTensorListFn, foreach_add_tensorlist_dispatcher)

using ForeachMulTensorListFn = void (*)(at::TensorList, at::TensorList);
DECLARE_DISPATCHER(ForeachMulTensorListFn, foreach_mul_tensorlist_dispatcher)

// --- Non-inplace ops (return vector<Tensor>) ---

using ForeachSqrtFn = ::std::vector<at::Tensor> (*)(at::TensorList);
DECLARE_DISPATCHER(ForeachSqrtFn, foreach_sqrt_dispatcher)

using ForeachNegFn = ::std::vector<at::Tensor> (*)(at::TensorList);
DECLARE_DISPATCHER(ForeachNegFn, foreach_neg_dispatcher)

using ForeachReciprocalFn = ::std::vector<at::Tensor> (*)(at::TensorList);
DECLARE_DISPATCHER(ForeachReciprocalFn, foreach_reciprocal_dispatcher)

} // namespace at::native::flagos
