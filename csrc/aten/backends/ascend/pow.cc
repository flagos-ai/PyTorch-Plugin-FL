// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../pow.h"
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor PowTensorScalarKernelAscend(const at::Tensor& self, const at::Scalar& exponent) {
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_exponent(exponent, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnPowTensorScalar, acl_self.get(), acl_exponent.get(), acl_out.get());
  return out;
}

FLAGOS_REGISTER_DISPATCH(PowTensorScalarFn, pow_tensor_scalar_stub, FlagosDevice::kAscend, PowTensorScalarKernelAscend)

} // namespace at::native::flagos
