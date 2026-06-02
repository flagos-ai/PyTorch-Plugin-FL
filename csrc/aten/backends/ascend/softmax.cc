// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../softmax.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor SoftmaxKernelAscend(const at::Tensor& self, int64_t dim, bool half_to_float) {
  namespace ascend = at::native::flagos::ascend;

  auto out_dtype = half_to_float ? at::kFloat : self.scalar_type();
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnSoftmax, acl_self.get(), dim, acl_out.get());
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(SoftmaxFn, softmax_dispatcher, Backend::kAscend, SoftmaxKernelAscend)

} // namespace at::native::flagos
