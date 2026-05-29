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

  auto self_contig = self.is_privateuseone() ? self : self.to(out.options());
  auto mat2_contig = mat2.is_privateuseone()
      ? (mat2.scalar_type() == out.scalar_type() ? mat2 : mat2.to(out.scalar_type()))
      : mat2.to(out.options());

  ascend::AclTensorWrapper acl_self(self_contig);
  ascend::AclTensorWrapper acl_mat2(mat2_contig);
  ascend::AclTensorWrapper acl_out(out);

  // allow_hf32=true: use fp32 accumulation for fp16 inputs, matching CUDA TensorCore behavior
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);
  EXEC_ASCEND_CMD(aclnnBatchMatMul, acl_self.get(), acl_mat2.get(), acl_out.get(), cube_math_type);
}

REGISTER_IMPL_TO_DISPATCHER(BmmFn, bmm_dispatcher, Backend::kAscend, BmmKernelAscend)

} // namespace at::native::flagos
