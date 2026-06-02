// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../constant_pad_nd.h"
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor ConstantPadNdKernelAscend(const at::Tensor& self, at::IntArrayRef pad,
                                     const at::Scalar& value) {
  namespace ascend = at::native::flagos::ascend;

  auto input_sizes = self.sizes().vec();
  auto ndim = input_sizes.size();
  auto pad_size = pad.size();

  std::vector<int64_t> out_sizes(input_sizes.begin(), input_sizes.end());
  for (size_t i = 0; i < pad_size / 2; ++i) {
    auto dim = ndim - 1 - i;
    out_sizes[dim] += pad[2 * i] + pad[2 * i + 1];
  }

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclIntArrayWrapper acl_pad(pad);
  ascend::AclScalarWrapper acl_value(value, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnConstantPadNd, acl_self.get(), acl_pad.get(),
                  acl_value.get(), acl_out.get());
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(ConstantPadNdFn, constant_pad_nd_dispatcher, Backend::kAscend, ConstantPadNdKernelAscend)

} // namespace at::native::flagos
