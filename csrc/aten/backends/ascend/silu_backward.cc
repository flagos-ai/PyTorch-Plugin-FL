// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../silu_backward.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor SiluBackwardKernelAscend(const at::Tensor& grad_output, const at::Tensor& self) {
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_grad_output(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_grad_input(grad_input);

  EXEC_ASCEND_CMD(aclnnSiluBackward, acl_grad_output.get(), acl_self.get(), acl_grad_input.get());
  return grad_input;
}

FLAGOS_REGISTER_DISPATCH(SiluBackwardFn, silu_backward_stub, FlagosDevice::kAscend, SiluBackwardKernelAscend)

} // namespace at::native::flagos
