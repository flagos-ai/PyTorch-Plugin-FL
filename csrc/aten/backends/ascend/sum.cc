// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../sum.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor SumDimKernelAscend(const at::Tensor& self, at::OptionalIntArrayRef dim,
                              bool keepdim, std::optional<at::ScalarType> dtype) {
  namespace ascend = at::native::flagos::ascend;

  auto out_dtype = dtype.has_value() ? dtype.value() : self.scalar_type();
  auto out_shape = self.sizes().vec();

  if (dim.has_value()) {
    auto dims = dim.value();
    std::vector<int64_t> sorted_dims(dims.begin(), dims.end());
    std::sort(sorted_dims.rbegin(), sorted_dims.rend());

    for (int64_t d : sorted_dims) {
      if (keepdim) {
        out_shape[d] = 1;
      } else {
        out_shape.erase(out_shape.begin() + d);
      }
    }
  } else {
    out_shape = keepdim ? std::vector<int64_t>(self.dim(), 1) : std::vector<int64_t>{};
  }

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  at::IntArrayRef dim_array = dim.has_value() ? dim.value() : at::IntArrayRef{};
  ascend::AclIntArrayWrapper acl_dim(dim_array);

  EXEC_ASCEND_CMD(aclnnSum, acl_self.get(), acl_dim.get(), keepdim,
                  static_cast<int8_t>(out_dtype), acl_out.get());
  return out;
}

FLAGOS_REGISTER_DISPATCH(SumDimFn, sum_dim_stub, FlagosDevice::kAscend, SumDimKernelAscend)

} // namespace at::native::flagos
