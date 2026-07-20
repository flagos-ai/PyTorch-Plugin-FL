// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

namespace {

// Core aclnn batch-matmul: self @ mat2 -> out (out already allocated & shaped).
void BmmIntoOut(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
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

}  // namespace

// Functional: bmm(self, mat2) -> Tensor. self:[b,n,k], mat2:[b,k,p] -> [b,n,p].
at::Tensor BmmKernelAscend(const at::Tensor& self, const at::Tensor& mat2) {
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      {self.size(0), self.size(1), mat2.size(2)}, self.options());
  BmmIntoOut(self, mat2, out);
  return out;
}

// Out variant: bmm.out(self, mat2, out=out) -> out&.
at::Tensor& BmmOutKernelAscend(
    const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  BmmIntoOut(self, mat2, out);
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(BmmFn, bmm_dispatcher, Backend::kAscend, BmmKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(BmmOutFn, bmm_out_dispatcher, Backend::kAscend, BmmOutKernelAscend)

} // namespace at::native::flagos
