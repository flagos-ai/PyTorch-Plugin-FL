// Copyright (c) 2026, BAAI. All rights reserved.

#include "foreach_ops.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(ForeachMulScalarFn, foreach_mul_scalar_dispatcher, "_foreach_mul_.Scalar")
ADD_IMPL_TO_DISPATCHER(ForeachAddScalarFn, foreach_add_scalar_dispatcher, "_foreach_add_.Scalar")
ADD_IMPL_TO_DISPATCHER(ForeachAddcdivScalarListFn, foreach_addcdiv_scalarlist_dispatcher, "_foreach_addcdiv_.ScalarList")
ADD_IMPL_TO_DISPATCHER(ForeachAddcmulScalarFn, foreach_addcmul_scalar_dispatcher, "_foreach_addcmul_.Scalar")
ADD_IMPL_TO_DISPATCHER(ForeachLerpScalarFn, foreach_lerp_scalar_dispatcher, "_foreach_lerp_.Scalar")
ADD_IMPL_TO_DISPATCHER(ForeachDivScalarListFn, foreach_div_scalarlist_dispatcher, "_foreach_div_.ScalarList")
ADD_IMPL_TO_DISPATCHER(ForeachAddTensorListFn, foreach_add_tensorlist_dispatcher, "_foreach_add_.List")
ADD_IMPL_TO_DISPATCHER(ForeachMulTensorListFn, foreach_mul_tensorlist_dispatcher, "_foreach_mul_.List")
ADD_IMPL_TO_DISPATCHER(ForeachSqrtFn, foreach_sqrt_dispatcher, "_foreach_sqrt")
ADD_IMPL_TO_DISPATCHER(ForeachNegFn, foreach_neg_dispatcher, "_foreach_neg")
ADD_IMPL_TO_DISPATCHER(ForeachReciprocalFn, foreach_reciprocal_dispatcher, "_foreach_reciprocal")

} // namespace at::native::flagos
