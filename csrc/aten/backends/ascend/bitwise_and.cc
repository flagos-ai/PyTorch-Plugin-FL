// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../bitwise_and.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor BitwiseAndTensorKernelAscend(const at::Tensor& self, const at::Tensor& other) {
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_other(other);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnBitwiseAnd, acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}

FLAGOS_REGISTER_DISPATCH(BitwiseAndTensorFn, bitwise_and_tensor_stub, FlagosDevice::kAscend, BitwiseAndTensorKernelAscend)

} // namespace at::native::flagos
