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
  int64_t ndim = self.dim();

  // Normalize dims to positive values
  std::vector<int64_t> norm_dims;
  if (dim.has_value() && !dim.value().empty()) {
    for (int64_t d : dim.value()) {
      norm_dims.push_back(d < 0 ? d + ndim : d);
    }
  } else {
    // When dim is not specified or empty, reduce over all dimensions
    for (int64_t d = 0; d < ndim; ++d) {
      norm_dims.push_back(d);
    }
  }

  // Compute output shape
  auto out_shape = self.sizes().vec();
  std::vector<int64_t> sorted_dims(norm_dims);
  std::sort(sorted_dims.rbegin(), sorted_dims.rend());
  for (int64_t d : sorted_dims) {
    if (keepdim) {
      out_shape[d] = 1;
    } else {
      out_shape.erase(out_shape.begin() + d);
    }
  }

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  ascend::AclIntArrayWrapper acl_dim(norm_dims);

  aclDataType acl_dtype = ascend::ToAclDataType(out_dtype);

  EXEC_ASCEND_CMD(aclnnReduceSum, acl_self.get(), acl_dim.get(), keepdim,
                  acl_dtype, acl_out.get());
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(SumDimFn, sum_dim_dispatcher, Backend::kAscend, SumDimKernelAscend)

} // namespace at::native::flagos
