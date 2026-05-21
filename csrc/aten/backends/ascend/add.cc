// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../add.h"

#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor AddKernelAscend(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  namespace ascend = at::native::flagos::ascend;

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_other(other);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());

  EXEC_ASCEND_CMD(aclnnAdd,
      acl_self.get(), acl_other.get(), acl_alpha.get(), acl_out.get());

  return out;
}

FLAGOS_REGISTER_DISPATCH(AddTensorFn, add_tensor_stub, FlagosDevice::kAscend, AddKernelAscend)

} // namespace at::native::flagos
