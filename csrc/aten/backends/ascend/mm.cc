// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mm.h"

#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

void MmKernelAscend(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  namespace ascend = at::native::flagos::ascend;
  // allow_hf32=true: use fp32 accumulation for fp16 inputs, matching CUDA TensorCore behavior
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_mat2(mat2);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnMm, acl_self.get(), acl_mat2.get(), acl_out.get(), cube_math_type);
}

REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kAscend, MmKernelAscend)

} // namespace at::native::flagos
