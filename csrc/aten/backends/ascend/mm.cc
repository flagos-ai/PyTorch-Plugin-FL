// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"

#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

namespace {

// Core aclnn matmul: writes self @ mat2 into a pre-allocated out tensor.
void MmComputeAscend(
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

} // namespace

// Functional variant: aten::mm(self, mat2) -> Tensor
at::Tensor MmKernelAscend(const at::Tensor& self, const at::Tensor& mat2) {
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      {self.size(0), mat2.size(1)}, self.options());
  MmComputeAscend(self, mat2, out);
  return out;
}

// Out variant: aten::mm.out(self, mat2, *, out) -> Tensor&
at::Tensor& MmOutKernelAscend(
    const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  MmComputeAscend(self, mat2, out);
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kAscend, MmKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(MmOutFn, mm_out_dispatcher, Backend::kAscend, MmOutKernelAscend)

} // namespace at::native::flagos
