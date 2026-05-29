// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../acos.h"

#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor AcosKernelAscend(const at::Tensor& self) {
  namespace ascend = at::native::flagos::ascend;

  // aclnnAcos does not support float64; cast down to float32 and back
  bool needs_cast = (self.scalar_type() == at::kDouble);
  auto input = needs_cast ? self.to(at::kFloat) : self;

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());

  ascend::AclTensorWrapper acl_self(input);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnAcos, acl_self.get(), acl_out.get());

  if (needs_cast) {
    // Device doesn't support float64 storage, so create a new float64 tensor
    // on the device and copy the float32 result into it
    auto out_f64 = at::empty_like(self);
    out_f64.copy_(out);
    return out_f64;
  }
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(AcosFn, acos_dispatcher, Backend::kAscend, AcosKernelAscend)

} // namespace at::native::flagos
