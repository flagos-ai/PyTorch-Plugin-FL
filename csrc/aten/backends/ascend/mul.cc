// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mul.h"
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor MulTensorKernelAscend(const at::Tensor& self, const at::Tensor& other) {
  namespace ascend = at::native::flagos::ascend;

  auto result_dtype = self.scalar_type();
  auto self_contig = self.is_privateuseone() ? self : self.to(self.options().device(c10::DeviceType::PrivateUse1));
  auto other_contig = other.is_privateuseone()
      ? (other.scalar_type() == result_dtype ? other : other.to(result_dtype))
      : other.to(self.options());

  auto out_shape = at::infer_size(self_contig.sizes(), other_contig.sizes());
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self_contig.options());

  ascend::AclTensorWrapper acl_self(self_contig);
  ascend::AclTensorWrapper acl_other(other_contig);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnMul, acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}

FLAGOS_REGISTER_DISPATCH(MulTensorFn, mul_tensor_stub, FlagosDevice::kAscend, MulTensorKernelAscend)

} // namespace at::native::flagos
