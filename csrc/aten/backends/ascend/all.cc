// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../all.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor AllKernelAscend(const at::Tensor& self) {
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      {}, self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnAll, acl_self.get(), acl_out.get());
  return out;
}

FLAGOS_REGISTER_DISPATCH(AllFn, all_stub, FlagosDevice::kAscend, AllKernelAscend)

} // namespace at::native::flagos
