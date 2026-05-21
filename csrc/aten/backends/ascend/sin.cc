// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../sin.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor SinKernelAscend(const at::Tensor& self) {
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnSin, acl_self.get(), acl_out.get());
  return out;
}

FLAGOS_REGISTER_DISPATCH(SinFn, sin_stub, FlagosDevice::kAscend, SinKernelAscend)

} // namespace at::native::flagos
