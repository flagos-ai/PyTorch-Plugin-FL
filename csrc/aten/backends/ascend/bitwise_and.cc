// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../bitwise_and.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor BitwiseAndTensorKernelAscend(const at::Tensor& self, const at::Tensor& other) {
  namespace ascend = at::native::flagos::ascend;

  auto out_shape = at::infer_size(self.sizes(), other.sizes());

  // aclnnBitwiseAndTensor does not support bool; cast to int8, compute, cast back
  bool is_bool = (self.scalar_type() == at::kBool);
  auto work_dtype = is_bool ? at::kChar : self.scalar_type();

  auto self_work = is_bool ? self.to(work_dtype) : self;
  auto other_work = is_bool ? other.to(work_dtype) : other;

  auto self_expanded = self_work.expand(out_shape).contiguous();
  auto other_expanded = other_work.expand(out_shape).contiguous();

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self_expanded.options());

  ascend::AclTensorWrapper acl_self(self_expanded);
  ascend::AclTensorWrapper acl_other(other_expanded);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnBitwiseAndTensor, acl_self.get(), acl_other.get(), acl_out.get());

  return is_bool ? out.to(at::kBool) : out;
}

FLAGOS_REGISTER_DISPATCH(BitwiseAndTensorFn, bitwise_and_tensor_stub, FlagosDevice::kAscend, BitwiseAndTensorKernelAscend)

} // namespace at::native::flagos
