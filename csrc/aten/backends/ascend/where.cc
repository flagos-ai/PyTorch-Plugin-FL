// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../where.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor WhereSelfKernelAscend(const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other) {
  namespace ascend = at::native::flagos::ascend;

  auto out_shape = at::infer_size(self.sizes(), other.sizes());
  out_shape = at::infer_size(condition.sizes(), out_shape);

  auto self_expanded = self.expand(out_shape).contiguous();
  auto other_expanded = other.expand(out_shape).contiguous();
  auto cond_expanded = condition.expand(out_shape).contiguous();

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  ascend::AclTensorWrapper acl_condition(cond_expanded);
  ascend::AclTensorWrapper acl_self(self_expanded);
  ascend::AclTensorWrapper acl_other(other_expanded);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnSWhere, acl_condition.get(), acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(WhereSelfFn, where_self_dispatcher, Backend::kAscend, WhereSelfKernelAscend)

} // namespace at::native::flagos
