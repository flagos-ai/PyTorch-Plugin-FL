// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../all.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor AllKernelAscend(const at::Tensor& self) {
  namespace ascend = at::native::flagos::ascend;

  // CANN 8.5 aclnnAll signature: (self, dim, keepdim, out)
  // Flatten to 1-D and reduce along dim=0.
  auto input = self.contiguous().reshape({-1});
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      {}, self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(input);
  ascend::AclTensorWrapper acl_out(out);

  int64_t dim_val = 0;
  at::IntArrayRef dim_arr(&dim_val, 1);
  ascend::AclIntArrayWrapper acl_dim(dim_arr);

  EXEC_ASCEND_CMD(aclnnAll, acl_self.get(), acl_dim.get(), false, acl_out.get());
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(AllFn, all_dispatcher, Backend::kAscend, AllKernelAscend)

} // namespace at::native::flagos
