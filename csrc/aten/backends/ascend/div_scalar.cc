// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../div_scalar.h"
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor DivScalarKernelAscend(const at::Tensor& self, const at::Scalar& other) {
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_other(other, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnDivs, acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}

FLAGOS_REGISTER_DISPATCH(DivScalarFn, div_scalar_stub, FlagosDevice::kAscend, DivScalarKernelAscend)

} // namespace at::native::flagos
