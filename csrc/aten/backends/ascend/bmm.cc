// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../bmm.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

void BmmKernelAscend(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  namespace ascend = at::native::flagos::ascend;
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(false);

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_mat2(mat2);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnBmm, acl_self.get(), acl_mat2.get(), acl_out.get(), cube_math_type);
}

FLAGOS_REGISTER_DISPATCH(BmmFn, bmm_stub, FlagosDevice::kAscend, BmmKernelAscend)

} // namespace at::native::flagos
